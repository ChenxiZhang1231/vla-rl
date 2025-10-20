import os, sys, subprocess, shutil

print("PY:", sys.executable)
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))

so_path = "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/miniconda3/envs/lrm_wm_simplerenv/lib/python3.10/site-packages/optree/_C.cpython-310-x86_64-linux-gnu.so"
print("\nldd of optree _C.so:")
try:
    out = subprocess.check_output(["ldd", so_path], text=True)
    print(out)
except Exception as e:
    print("ldd failed:", e)

print("\nldconfig libstdc++.so.6 candidates (top 5):")
try:
    out = subprocess.check_output("ldconfig -p | grep 'libstdc++.so.6' | head -5", shell=True, text=True)
    print(out)
except Exception as e:
    print("ldconfig failed:", e)

# 尝试找到 64-bit libstdc++.so.6 并打印 GLIBCXX 符号（若系统有 strings/ldconfig）
ldconfig = shutil.which("ldconfig")
strings = shutil.which("strings")
if ldconfig and strings:
    try:
        path = subprocess.check_output(
            "ldconfig -p | grep 'libstdc++.so.6 (64-bit)' | head -1 | awk '{print $4}'",
            shell=True, text=True).strip()
        if path:
            print("\nGLIBCXX symbols in:", path)
            out = subprocess.check_output([strings, path], text=True)
            # 只显示结尾几十行，看看是否包含 3.4.31
            lines = [l for l in out.splitlines() if l.startswith("GLIBCXX_")]
            print("\n".join(lines[-30:]))
        else:
            print("\nNo 64-bit libstdc++.so.6 found via ldconfig.")
    except Exception as e:
        print("strings check failed:", e)
else:
    print("\nSkip strings/ldconfig symbol check (tool missing).")
