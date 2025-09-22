import cv2
from PIL import Image
import time

# 使用 OpenCV 保存图片
def save_with_cv2(image, path):
    start_time = time.time()
    cv2.imwrite(path, image)
    end_time = time.time()
    return end_time - start_time

# 使用 Pillow 保存图片
def save_with_pillow(image, path):
    start_time = time.time()
    image.save(path)
    end_time = time.time()
    return end_time - start_time

# 加载图像
left_image_cv2 = cv2.imread("left_image_with_depth_projection.png")
left_image_pillow = Image.open("left_image_with_depth_projection.png")

# 比较保存时间
cv2_time = save_with_cv2(left_image_cv2, "cv2_output.jpg")
pillow_time = save_with_pillow(left_image_pillow, "pillow_output.jpg")

print(f"OpenCV 保存时间: {cv2_time:.4f} 秒")
print(f"Pillow 保存时间: {pillow_time:.4f} 秒")
