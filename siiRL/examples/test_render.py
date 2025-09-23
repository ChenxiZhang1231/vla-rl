import mujoco
import numpy as np
from PIL import Image
import os

# 增加对 PyOpenGL 的依赖，用于检查渲染器供应商
try:
    from OpenGL import GL
except ImportError as e:
    print("错误：缺少 PyOpenGL 模块。请先运行 'pip install PyOpenGL'")
    exit()

# 设置环境变量，确保使用 EGL 进行无头渲染
# 最好在运行脚本前在终端设置，但在这里设置也可以作为一种检查
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# 一个简单的 MuJoCo XML 模型
xml = """
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

renderer = None
while True:
    try:
        print("正在加载 MuJoCo 模型...")
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        
        print("创建渲染器 (Renderer)...")
        # 创建一个渲染器，这是进行 GPU 渲染的关键
        # 如果驱动或 EGL 配置不正确，通常会在这里失败
        renderer = mujoco.Renderer(model, height=480, width=640)
        
        # --- 新增诊断代码 ---
        # 获取并打印 OpenGL 供应商信息
        # 这可以明确告诉我们当前是 NVIDIA 驱动在工作还是 Mesa 在工作
        vendor = GL.glGetString(GL.GL_VENDOR)
        print(f"检测到 OpenGL 供应商: {vendor.decode('utf-8')}")
        if b'NVIDIA' not in vendor:
            print("\n!!! 警告：未检测到 NVIDIA 渲染器。!!!")
            print("当前可能正在使用 CPU 进行软件渲染。请检查 NVIDIA 驱动和 EGL 配置。\n")
        else:
            print("成功检测到 NVIDIA 渲染器！")
        # --- 诊断代码结束 ---

        print("更新场景并渲染...")
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        
        # 获取渲染后的图像
        pixels = renderer.render()
        
        print("渲染成功！图像尺寸:", pixels.shape)
        
        # 保存图像
        img = Image.fromarray(pixels)
        img.save('test_render.png')
        
        print("测试图像已保存为 'test_render.png'。")

    except Exception as e:
        print("\n--- 渲染测试失败 ---")
        print("错误类型:", type(e).__name__)
        print("错误信息:", e)
        print("\n请检查以下几点：")
        print("1. NVIDIA 驱动是否已正确安装？ (运行 `nvidia-smi` 查看)")
        print("2. `MUJOCO_GL=egl` 和 `PYOPENGL_PLATFORM=egl` 环境变量是否已设置？")
        print("3. 是否安装了所有必要的依赖库，如 `libglew-dev`, `libosmesa6-dev`, `patchelf` 等？")
        print("4. 如果在容器中运行，是否已正确配置 Docker 的 GPU 支持？")

    finally:
        # 清理资源
        if renderer:
            renderer.close()
            print("渲染器已关闭。")
