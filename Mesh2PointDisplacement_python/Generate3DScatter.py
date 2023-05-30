# 在0-255的3D范围随机生成散射点

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义散点的数量
num_points = 1000

# 在0-255的范围内生成随机的x，y，z坐标
x = np.random.randint(0, 256, num_points)
y = np.random.randint(0, 256, num_points)
z = np.random.randint(0, 256, num_points)

# 生成随机的灰度值
colors = np.random.rand(num_points)

# 创建一个三维的绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 使用scatter函数绘制散点
sc = ax.scatter(x, y, z, c=colors, cmap='gray')

# 显示图像
plt.show()
