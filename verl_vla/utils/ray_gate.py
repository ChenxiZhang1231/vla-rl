import time, threading, queue, uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor

import ray
from swift.plugin import InferStats
from swift.llm.infer.protocol import RequestConfig, InferRequest
from swift.llm.infer.infer_engine.infer_client import InferClient

@dataclass
class _Job:
    key: str
    req: InferRequest

@ray.remote(num_cpus=0)
class RMGateway:
    def __init__(
        self,
        endpoints: List[Tuple[str, int]],
        max_batch_size: int = 512,
        max_wait_ms: int = 8,
        default_cfg: Optional[dict] = None,
        round_robin: bool = True,
        queue_maxsize: int = 10000,
        # 新增：并发相关
        concurrency: int = None,      # 线程池并发度，默认=len(endpoints)
        max_inflight: int = None,     # 允许同时在飞的批数，默认=concurrency
    ):
        self.endpoints = endpoints
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.default_cfg = default_cfg or {"temperature": 0.0, "max_tokens": 64, "stream": False}
        self.round_robin = round_robin

        # 缓存客户端，避免频繁创建
        self._clients = [InferClient(host=h, port=p) for (h, p) in endpoints]
        self._idx = -1  # 让第一次 pick 到 0

        self.metric = InferStats()
        self.q: "queue.Queue[_Job]" = queue.Queue(maxsize=queue_maxsize)
        self.results: Dict[str, object] = {}
        self._res_lock = threading.Lock()
        self._clients = [InferClient(host=h, port=p) for (h, p) in endpoints]
        self._alive = [True] * len(self._clients)
        self._idx = -1
        self._alive_lock = threading.Lock()
        self._inflight_lock = threading.Lock()
        self._inflight_count = 0

        # 并发控制
        if concurrency is None:
            concurrency = max(1, len(self._clients))
        if max_inflight is None:
            max_inflight = concurrency
        self._executor = ThreadPoolExecutor(max_workers=concurrency)
        self._inflight_sem = threading.Semaphore(max_inflight)

        self._stop = threading.Event()
        self._t = threading.Thread(target=self._flush_loop, daemon=True)
        self._t.start()

    # ---------- 对外 API ----------
    def submit(self, req: InferRequest) -> str:
        k = uuid.uuid4().hex
        self.q.put(_Job(key=k, req=req), block=True)
        return k

    def submit_many(self, reqs: List[InferRequest]) -> List[str]:
        keys = []
        for r in reqs:
            k = uuid.uuid4().hex
            self.q.put(_Job(key=k, req=r), block=True)
            keys.append(k)
        return keys

    def get_result_blocking(self, key: str, timeout_s: Optional[float] = None):
        start = time.time()
        while True:
            with self._res_lock:
                res = self.results.pop(key, None)
            if res is not None:
                if isinstance(res, Exception):
                    raise res
                return res
            if timeout_s is not None and time.time() - start > timeout_s:
                raise TimeoutError(f"timeout waiting result for key={key}")
            time.sleep(0.002)

    def get_many_blocking(self, keys: List[str], timeout_s: Optional[float] = None):
        start = time.time()
        results = [None] * len(keys)
        remaining = set(range(len(keys)))
        while remaining:
            with self._res_lock:
                for i in list(remaining):
                    k = keys[i]
                    res = self.results.pop(k, None)
                    if res is not None:
                        results[i] = res
                        remaining.remove(i)
            if remaining:
                if timeout_s is not None and time.time() - start > timeout_s:
                    missing = [keys[i] for i in sorted(remaining)]
                    raise TimeoutError(f"timeout waiting keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
                time.sleep(0.002)
        # 如果有批超时被置为 TimeoutError，这里会直接返回异常对象；你可以在上层检查并决定是否重试
        for r in results:
            if isinstance(r, Exception):
                # 也可以 raise，第一个异常
                raise r
        return results

    def stop(self):
        self._stop.set()
        self._t.join(timeout=1.0)
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ---------- 内部逻辑 ----------
    def _pick_client_idx(self) -> int:
        with self._alive_lock:
            n = len(self._clients)
            for _ in range(n):
                self._idx = (self._idx + 1) % n
                if self._alive[self._idx]:
                    return self._idx
            # 如果都不可用，就先默认回 0（后面会报错写回）
            return 0

    def _mark_dead(self, idx: int):
        with self._alive_lock:
            self._alive[idx] = False

    def _mark_alive(self, idx: int):
        with self._alive_lock:
            self._alive[idx] = True

    def _dispatch_batch_async(self, jobs: List[_Job], serve_timeout_s: float = 300.0, retry: int = 1):
        client_idx = self._pick_client_idx()
        client = self._clients[client_idx]
        cfg = RequestConfig(**self.default_cfg)
        infer_reqs = [j.req for j in jobs]
        batch_keys = [j.key for j in jobs]

        done_flag = threading.Event()

        def _write_results(obj):
            with self._res_lock:
                for k, v in zip(batch_keys, obj if isinstance(obj, list) else [obj] * len(batch_keys)):
                    # 不覆盖已有（例如超时兜底先写过）
                    if k not in self.results:
                        self.results[k] = v

        def _work_once(c_idx, clin):
            t0 = time.time()
            try:
                outs = clin.infer(infer_reqs, request_config=cfg, metrics=[self.metric])
                self._mark_alive(c_idx)
                _write_results(outs)
                print(f"[GW] done endpoint={c_idx} size={len(jobs)} elapsed={time.time()-t0:.3f}s")
                return True
            except Exception as e:
                self._mark_dead(c_idx)
                _write_results(e)
                print(f"[GW] ERROR endpoint={c_idx} size={len(jobs)} exc={e}")
                return False

        def _watchdog():
            # 兜底：超过 serve_timeout_s，把剩余未写的 key 置为 TimeoutError
            if not done_flag.wait(timeout=serve_timeout_s):
                with self._res_lock:
                    for k in batch_keys:
                        if k not in self.results:
                            self.results[k] = TimeoutError(f"rm batch timeout {serve_timeout_s}s")
                print(f"[GW] TIMEOUT endpoint={client_idx} size={len(jobs)} >{serve_timeout_s}s")

        def _work():
            try:
                # 先启一个 watchdog
                wd = threading.Thread(target=_watchdog, daemon=True)
                wd.start()

                ok = _work_once(client_idx, client)
                if (not ok) and retry > 0:
                    # 失败重试一次：换下一个活着的端点
                    alt_idx = self._pick_client_idx()
                    alt_client = self._clients[alt_idx]
                    _work_once(alt_idx, alt_client)
            finally:
                done_flag.set()
                with self._inflight_lock:
                    self._inflight_count -= 1
                self._inflight_sem.release()

        with self._inflight_lock:
            self._inflight_count += 1
        self._executor.submit(_work)

    def _flush_loop(self):
        pending: List[_Job] = []
        last_ts = time.time()
        max_wait_s = self.max_wait_ms / 1000.0

        while not self._stop.is_set():
            # A. 无阻塞尽量捞活
            try:
                while True:
                    pending.append(self.q.get_nowait())
            except queue.Empty:
                pass

            # B. 只要 pending 足够或窗口到了，就连发多批，直到 in-flight 满
            dispatched = False
            while pending and (len(pending) >= self.max_batch_size or (time.time() - last_ts) >= max_wait_s):
                if not self._inflight_sem.acquire(blocking=False):
                    break  # 并发满了，下轮再发
                batch = pending[:self.max_batch_size]
                del pending[:self.max_batch_size]
                self._dispatch_batch_async(batch)   # 异步发出
                last_ts = time.time()
                dispatched = True

            if dispatched:
                continue

            # C. 如果 pending 不足以成批/并发已满：短暂等待新活或窗口
            timeout = max(max_wait_s - (time.time() - last_ts), 0)
            try:
                job = self.q.get(timeout=timeout if not pending else 0.0)
                pending.append(job)
            except queue.Empty:
                pass