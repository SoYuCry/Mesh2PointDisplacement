
import numpy as np

# 创建四面体的顶点坐标  
vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

# 创建点  
point = [0.23, 0.34, 0.1]

# 判断点是否在四面体内  
def is_in_tetrahedron(Point, vertices,Vol):  
    alpha = calculateVolum(vertices[0],vertices[1],vertices[2],Point)
    beta = calculateVolum(vertices[0],vertices[2],vertices[3],Point)
    gamma = calculateVolum(vertices[0],vertices[1],vertices[3],Point)
    delta = calculateVolum(vertices[1],vertices[2],vertices[3],Point)
    if (abs(Vol-(alpha+beta+gamma+delta)<1e-15)):  
        print("点在四面体内")  
    else:  
        print("点不在四面体内")  

    return Vol==(alpha+beta+gamma+delta)


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

# 检查点是否在四面体内  
vol = calculateVolum(vertices[0],vertices[1],vertices[2],vertices[3])
is_in_tetrahedron_result = is_in_tetrahedron(point, vertices,vol)  