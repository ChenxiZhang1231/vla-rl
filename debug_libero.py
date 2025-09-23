from libero.libero import benchmark
from verl_vla.utils.libero_utils import get_libero_env, get_libero_dummy_action, get_image_resize_size, get_libero_image, get_libero_wrist_image, quat2axisangle, normalize_gripper_action, invert_gripper_action, save_rollout_video

import time 
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict['libero_spatial']()
task = task_suite.get_task(0)
initial_states = task_suite.get_task_init_states(0)
initial_state = initial_states[0]

env, task_description = get_libero_env(task, 'openvla', resolution=256)
env.reset()
obs = env.set_init_state(initial_state)

for _ in range(50):
    obs, _, _, _ = env.step(get_libero_dummy_action('openvla'))
    
start_time = time.time()
for _ in range(50):
    obs, _, _, _ = env.step(get_libero_dummy_action('openvla'))
end_time = time.time() - start_time
print(end_time)
