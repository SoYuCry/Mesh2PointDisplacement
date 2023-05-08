import numpy as np
# 1、Read .STL
from stl import mesh
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits import mplot3d

# 导出顶点坐标和顶点坐标处XYZ方向位移
def displacement_list(TXTname):
    # 读取txt文件,导出每行中的第四项
    displacement_list = []
    with open(TXTname, 'r') as f:  
        lines = f.readlines()   

    # 从第八行开始是顶点坐标部分
    for i in range(8,len(lines)):
        displacement_list.append(float(lines[i].split()[3]))
    return displacement_list

def points_mesh_list(TXTname):
    points_mesh = []
    with open(TXTname, 'r') as f:  
        lines = f.readlines()
    for i in range(8,len(lines)):
        point_mesh = [float(lines[i].split()[0]),float(lines[i].split()[1]),float(lines[i].split()[2])]
        points_mesh.append(point_mesh)
    return points_mesh


def Creat_Scatter():
    points = []
    k,j,i = [0]*3
    while k < XRANGE:
        while j < YRANGE:
            while i < ZRANGE:
                points.append([k,j,i])
                i = i + ZSTEP
            j = j + YSTEP
            i = 0
        k = k + XSTEP
        j = 0
    return points

def Mesh2DT(points_mesh):
    x = []
    y = []
    z = []
    for i in range(len(points_mesh)):
        x.append([points_mesh[i][0]])
        y.append([points_mesh[i][1]])
        z.append([points_mesh[i][2]])
    x=np.array(x)
    y=np.array(y)
    z=np.array(z)

    # 顶点的array数组
    points_forDT = np.hstack([x, y, z])

    DT = Delaunay(points_forDT)
    return DT

def getPoint_mesh_new():
    points_mesh_new = []
    x_displacement = displacement_list('T_end_x.txt')
    y_displacement = displacement_list('T_end_y.txt')
    z_displacement = displacement_list('T_end_z.txt')
    for i in range(len(points_mesh)):
        points_mesh_new.append([points_mesh[i][0] + x_displacement[i], points_mesh[i][1] + y_displacement[i], points_mesh[i][2] + z_displacement[i]])

    return points_mesh_new

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

def is_in_tetrahedron(Point, vertices,Vol):  
    # alpha = calculateVolum(vertices[0],vertices[1],vertices[2],Point)
    # beta = calculateVolum(vertices[0],vertices[2],vertices[3],Point)
    # gamma = calculateVolum(vertices[0],vertices[1],vertices[3],Point)
    # delta = calculateVolum(vertices[1],vertices[2],vertices[3],Point)
    # # if (abs(Vol-(alpha+beta+gamma+delta))<1e-15):  
    # #     print("点在四面体内")  
    # # else:  
    # #     print("点不在四面体内")  

    # return ((abs(Vol-(alpha+beta+gamma+delta))<1e-15) & (Vol > 1e-15))

    # Create the Delaunay triangulation
    try:
        tetra = Delaunay(vertices.tolist())
    except:
        return(False)

    # Find the simplex (tetrahedron) containing the point
    simplex = tetra.find_simplex(Point)

    # Check if the point is inside the tetrahedron
    # if simplex != -1:
    #     print("The point is inside the tetrahedron.")
    # else:
    #     print("The point is outside the tetrahedron.")

    return(simplex != -1)

def relativePlace_index(vertices,Point,Vol):
    alpha = calculateVolum(vertices[0],vertices[1],vertices[2],Point)/Vol
    beta = calculateVolum(vertices[0],vertices[2],vertices[3],Point)/Vol
    gamma = calculateVolum(vertices[0],vertices[1],vertices[3],Point)/Vol
    delta = calculateVolum(vertices[1],vertices[2],vertices[3],Point)/Vol
    return [alpha,beta,gamma,delta]

def relativePlace_calculate(alpha,beta,gamma,delta,vertices):
    relativePoin = \
    [alpha*vertices[3][0]+beta*vertices[1][0]+gamma*vertices[2][0]+delta*vertices[0][0]\
    ,alpha*vertices[3][1]+beta*vertices[1][1]+gamma*vertices[2][1]+delta*vertices[0][1]\
    ,alpha*vertices[3][2]+beta*vertices[1][2]+gamma*vertices[2][2]+delta*vertices[0][2]]

    return relativePoin
 
def getNewPoints(tetra_vertices,tetra_vertices_new,points,points_standby):
    points_new = []
    # 遍历网格的四面体
    for i in range(0,len(tetra_vertices)):
        vertices = tetra_vertices[i]
        vertices_new = tetra_vertices_new[i]

        Vol = calculateVolum(vertices[0],vertices[1],vertices[2],vertices[3])
        for point in points:
            
            # showPlace(point,tetra_vertices[i])
            if (is_in_tetrahedron(point, vertices,Vol)):
                # print("该点在四面体中")
                # _ = is_in_tetrahedron(point, vertices,Vol)
                alpha,beta,gamma,delta = relativePlace_index(vertices,point,Vol)
                relativePoin = relativePlace_calculate(alpha,beta,gamma,delta,vertices_new)
                points_new.append(relativePoin)
                points.remove(point)
                points_standby.append(point)
            else:
                continue
    return points_new
# 这里决定了散射体的范围和密度

def tempPlot(point,vertices):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    ax.add_collection3d(Poly3DCollection(vertices[faces], alpha=0.25, facecolor='lightskyblue'))
    ax.scatter([point[0]], [point[1]], [point[2]], color='red', s=10, alpha=1, depthshade=False)
    ax.set_xlim(0,3)
    ax.set_ylim(0,3)
    ax.set_zlim(0,3)
    return
XRANGE = 3
XSTEP = 1 
YRANGE = 3
YSTEP = 1
ZRANGE = 3
ZSTEP = 1

# 构建散射体分布
points = Creat_Scatter()
# 备用的存放原始散射体位置的列表
points_standby = []

points_mesh = points_mesh_list('T_end_x.txt')
points_mesh_new = getPoint_mesh_new()

# 通过加xyz三个方向上的位移,存储得到变形后的Mesh的位置

# 对原始网格的顶点进行三角剖分
DT = Mesh2DT(points_mesh)


# 在DT数据结构中,提取得到四边形列表
points_mesh_new = np.array(points_mesh_new)
points_mesh = np.array(points_mesh)

# 按照三角剖分的结果细分为四面体
tetra_vertices = DT.points[DT.simplices]
tetra_vertices_new = points_mesh_new[DT.simplices]

# 计算新的散射体的坐标位置
points_new = getNewPoints(tetra_vertices,tetra_vertices_new,points,points_standby)

# 方便后续调整坐标轴
scale = points_mesh.flatten()
scale_new = points_mesh_new.flatten()


# 遍历网格的四面
fig = plt.figure()

ax = fig.add_subplot(121, projection='3d')
faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])

# 添加网格顶点
ax.scatter([p[0] for p in points_mesh], [p[1] for p in points_mesh], [p[2] for p in points_mesh],
           color='red', s=10, alpha=1, depthshade=False)
# 添加原始散射体
ax.scatter([p[0] for p in points_standby], [p[1] for p in points_standby], [p[2] for p in points_standby],
           color='green', s=5, alpha=1, depthshade=False)
# 可视化一些系列四面体
for i in range(0,len(tetra_vertices)):
    t = Poly3DCollection(tetra_vertices[i][faces], alpha=0.25, facecolor='lightskyblue')
    ax.add_collection3d(t)

# 设置坐标轴范围和标签,这里其实想写一个自动画范围的东西

ax.auto_scale_xyz(scale, scale, scale)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


#绘制变形后散点分布
ay = fig.add_subplot(122, projection='3d')



# 添加网格顶点
ay.scatter([p[0] for p in points_mesh_new], [p[1] for p in points_mesh_new], [p[2] for p in points_mesh_new],
           color='red', s=10, alpha=1, depthshade=False)
# 添加散射体
ay.scatter([p[0] for p in points_new], [p[1] for p in points_new], [p[2] for p in points_new],
           color='green', s=5, alpha=1, depthshade=False)
# 可视化一些系列四面体
for i in range(0,len(tetra_vertices_new)):
    t = Poly3DCollection(tetra_vertices_new[i][faces], alpha=0.25, facecolor='lightskyblue')
    ay.add_collection3d(t)


# 设置坐标轴范围和标签
# ay.set_xlim([0, 3])
# ay.set_ylim([0, 3])
# ay.set_zlim([0, 3])
# ay.auto_scale_xyz(scale_new, scale_new, scale_new)
ay.auto_scale_xyz(scale, scale, scale)
ay.set_xlabel('X')
ay.set_ylabel('Y')
ay.set_zlabel('Z')

plt.show() # 显示图像

pass