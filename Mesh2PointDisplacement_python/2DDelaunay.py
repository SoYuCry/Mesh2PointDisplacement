import numpy as np
points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
from scipy.spatial import Delaunay
tri = Delaunay(points)
 
# 画出来:

import matplotlib.pyplot as plt
plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
# tri.simplices里面存放了点的连接关系
plt.plot(points[:,0], points[:,1], 'o')
plt.show()