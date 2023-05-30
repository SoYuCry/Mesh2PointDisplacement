
# 临时用ChatGPT生成的用于生成2D散点图象的py

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 创建一个256x256的全0矩阵
image = np.zeros((256, 256))

# 定义散点的数量
num_points = 1000

# 生成随机散点的位置和灰度值
for _ in range(num_points):
    x = np.random.randint(0, 256)
    y = np.random.randint(0, 256)
    gray_value = np.random.rand()
    image[x, y] = gray_value

# 使用matplotlib来显示图像
plt.imshow(image, cmap='gray')
plt.show()

image_8bit = np.uint8(image * 255)

Image.fromarray(image_8bit).save('random_dots.png')