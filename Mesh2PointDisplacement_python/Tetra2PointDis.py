import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 计算四面体的体积
def calculateVolum(A,B,C,D):
    # 辅助矩阵 M
    M = np.matrix([[A[0], A[1], A[2], 1],
                [B[0], B[1], B[2], 1],
                [C[0], C[1], C[2], 1],
                [D[0], D[1], D[2], 1]])

    # 四面体体积的计算
    det_M = np.linalg.det(M)
    M1 = np.matrix(np.copy(M))
    M1[:, 0] = np.array([A[0], B[0], C[0], D[0]]).reshape((4,1))
    M1[:, 1] = np.array([A[1], B[1], C[1], D[1]]).reshape((4,1))
    M1[:, 2] = np.array([A[2], B[2], C[2], D[2]]).reshape((4,1))
    det_M1 = np.linalg.det(M1)
    volume = abs(det_M1) / 6.0

    return volume
# 计算相对位置系数
def relativePlace_index(vertices,Point,Vol):
    alpha = calculateVolum(vertices[0],vertices[1],vertices[2],Point)/Vol
    beta = calculateVolum(vertices[0],vertices[2],vertices[3],Point)/Vol
    gamma = calculateVolum(vertices[0],vertices[1],vertices[3],Point)/Vol
    delta = calculateVolum(vertices[1],vertices[2],vertices[3],Point)/Vol
    return [alpha,beta,gamma,delta]

# 通过系数计算新坐标
def relativePlace_calculate(alpha,beta,gamma,delta,vertices):
    relativePoin = \
    [alpha*vertices[3][0]+beta*vertices[1][0]+gamma*vertices[2][0]+delta*vertices[0][0]\
    ,alpha*vertices[3][1]+beta*vertices[1][1]+gamma*vertices[2][1]+delta*vertices[0][1]\
    ,alpha*vertices[3][2]+beta*vertices[1][2]+gamma*vertices[2][2]+delta*vertices[0][2]]
    return relativePoin

# 定义四面体四个顶点的坐标
vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
vertices_new = np.array([[0, 0, 0], [0.8, 0, 0], [0.2, 0.8, 0], [0.1, 0.1, 0.6]])
# 定义四个面的索引
faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
#计算总体积
vol = calculateVolum(vertices[0],vertices[1],vertices[2],vertices[3])
points = [[0.23, 0.34, 0.1],[0.1, 0.2, 0.56],[0.264, 0.367, 0.3425],[0.636, 0.2354, 0.1]]
newpoints = []
for point in points:
    alpha,beta,gamma,delta = relativePlace_index(vertices,point,vol)
    relativePoin = relativePlace_calculate(alpha,beta,gamma,delta,vertices_new)

    vol1 = calculateVolum(vertices_new[0],vertices_new[1],vertices_new[2],vertices_new[3])
    alpha1,beta1,gamma1,delta1 = relativePlace_index(vertices_new,relativePoin,vol1)



    newpoints.append(relativePoin)


# 创建3D图像对象和绘制原始散点分布
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')

# 创建Poly3DCollection对象，用于绘制四面体
t = Poly3DCollection(vertices[faces], alpha=0.25, facecolor='lightskyblue')
ax.add_collection3d(t)

# 添加坐标点
ax.scatter([p[0] for p in points], [p[1] for p in points], [p[2] for p in points],
           color='red', s=10, alpha=1, depthshade=False)

# 设置坐标轴范围和标签
ax.set_xlim([-0.5, 1.5])
ax.set_ylim([-0.5, 1.5])
ax.set_zlim([-0.5, 1.5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


#绘制变形后散点分布
ay = fig.add_subplot(122, projection='3d')
# 创建Poly3DCollection对象，用于绘制四面体
t = Poly3DCollection(vertices_new[faces], alpha=0.25, facecolor='lightskyblue')
ay.add_collection3d(t)

# 添加坐标点
ay.scatter([p[0] for p in newpoints], [p[1] for p in newpoints], [p[2] for p in newpoints],
           color='red', s=10, alpha=1, depthshade=False)

# 设置坐标轴范围和标签
ay.set_xlim([-0.5, 1.5])
ay.set_ylim([-0.5, 1.5])
ay.set_zlim([-0.5, 1.5])
ay.set_xlabel('X')
ay.set_ylabel('Y')
ay.set_zlabel('Z')

plt.show() # 显示图像