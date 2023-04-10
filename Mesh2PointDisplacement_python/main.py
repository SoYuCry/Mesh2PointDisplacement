import numpy as np
# 1、Read .STL
from stl import mesh
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits import mplot3d

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

# 判断点是否在四面体内  
def is_in_tetrahedron(Point, vertices,Vol):  
    alpha = calculateVolum(vertices[0],vertices[1],vertices[2],Point)
    beta = calculateVolum(vertices[0],vertices[2],vertices[3],Point)
    gamma = calculateVolum(vertices[0],vertices[1],vertices[3],Point)
    delta = calculateVolum(vertices[1],vertices[2],vertices[3],Point)
    # if (abs(Vol-(alpha+beta+gamma+delta))<1e-15):  
    #     print("点在四面体内")  
    # else:  
    #     print("点不在四面体内")  

    return abs(Vol-(alpha+beta+gamma+delta))<1e-15

# 通过系数计算新坐标
def relativePlace_calculate(alpha,beta,gamma,delta,vertices):
    relativePoin = \
    [alpha*vertices[3][0]+beta*vertices[1][0]+gamma*vertices[2][0]+delta*vertices[0][0]\
    ,alpha*vertices[3][1]+beta*vertices[1][1]+gamma*vertices[2][1]+delta*vertices[0][1]\
    ,alpha*vertices[3][2]+beta*vertices[1][2]+gamma*vertices[2][2]+delta*vertices[0][2]]

    return relativePoin

def is_unique(newpoint,points):
    flag = False
    for point in points:
        if newpoint == point:
            flag = True
            break
    return flag

def Mesh2DT(mesh):
    points = []
    # 由于顶点从三角片中提取,所以可能有重叠
    for i in range(len(mesh.points)):
        if(is_unique([mesh.points[i][0],mesh.points[i][1],mesh.points[i][2]],points)):
            pass
        else:
            points.append([mesh.points[i][0],mesh.points[i][1],mesh.points[i][2]])

        if(is_unique([mesh.points[i][3],mesh.points[i][4],mesh.points[i][5]],points)):
            pass
        else:
            points.append([mesh.points[i][3],mesh.points[i][4],mesh.points[i][5]])
            
        if(is_unique([mesh.points[i][6],mesh.points[i][7],mesh.points[i][8]],points)):
            pass
        else:
            points.append([mesh.points[i][6],mesh.points[i][7],mesh.points[i][8]])

    # 提取顶点的xyz坐标
    x = []
    y = []
    z = []

    for i in range(len(points)):
        x.append([points[i][0]])
        y.append([points[i][1]])
        z.append([points[i][2]])

    x=np.array(x)
    y=np.array(y)
    z=np.array(z)

    # 顶点的array数组

    points_forDT = np.hstack([x, y, z])
    DT = Delaunay(points_forDT)
    return DT,points



def showPlace(point,vertices):
    # 逐个检查每个四面体,可视化的方法看散点是否在里面
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 创建Poly3DCollection对象，用于绘制四面体
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    t = Poly3DCollection(vertices[faces], alpha=0.25, facecolor='lightskyblue')
    ax.add_collection3d(t)

    # 添加坐标点
    ax.scatter([point[0]], [point[1]], [point[2]],
            color='red', s=10, alpha=1, depthshade=False)

    # 设置坐标轴范围和标签
    ax.set_xlim(min(vertices[0][0],vertices[1][0],vertices[2][0]),max(vertices[0][0],vertices[1][0],vertices[2][0]))
    ax.set_ylim(min(vertices[0][1],vertices[1][1],vertices[2][1]),max(vertices[0][1],vertices[1][1],vertices[2][1]))
    ax.set_zlim(min(vertices[0][2],vertices[1][2],vertices[2][2]),max(vertices[0][2],vertices[1][2],vertices[2][2]))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show() # 显示图像

def displacement_list(TXTname):

    # 读取txt文件,导出每行中的第四项
    displacement_list = []
    with open(TXTname, 'r') as f:  
        lines = f.readlines()

    for i in range(8,len(lines)):
        displacement_list.append(float(lines[i].split()[3]))

    return(displacement_list)



# 读取变形前后.stl文件
your_mesh = mesh.Mesh.from_file('T0.stl')
# your_mesh_new = mesh.Mesh.from_file('x_surface_0.stl')

x_displacement = displacement_list('T10_x.txt')
y_displacement = displacement_list('T10_y.txt')
z_displacement = displacement_list('T10_z.txt')


# 散点列表
points = [[0.01, 0.01, 0.01],[0.02, 0.02, 0.019]]
# 计算后新的散点位置
points_new = []
# 通过加xyz三个方向上的位移,存储得到变形后的Mesh的位置
points_mesh_new = []

# 利用三角剖分得到对.STL文件提取出的顶点,划分好的N个四边形
DT,points_mesh = Mesh2DT(your_mesh)
# DT_new = Mesh2DT(your_mesh_new)

# 把每个点坐标点的位移加上，得到变形后的mesh坐标
for i in range(len(points_mesh)):
    points_mesh[i][0] +=  x_displacement[i]
    points_mesh[i][1] +=  y_displacement[i]
    points_mesh[i][2] +=  z_displacement[i]
    points_mesh_new.append(points_mesh[i])

# 在DT数据结构中,提取得到四边形列表
points_mesh_new = np.array(points_mesh_new)
tetra_vertices = DT.points[DT.simplices]
tetra_vertices_new = points_mesh_new[DT.simplices]

# 方便后续调整坐标轴
scale = your_mesh.points.flatten()
scale_new = points_mesh_new.flatten()


# 遍历网格的四面体
for i in range(0,len(tetra_vertices)):
    vertices = tetra_vertices[i]
    vertices_new = tetra_vertices_new[i]

    Vol = calculateVolum(vertices[0],vertices[1],vertices[2],vertices[3])
    for point in points:

        # showPlace(point,tetra_vertices[i])
        if (is_in_tetrahedron(point, vertices,Vol)):
            alpha,beta,gamma,delta = relativePlace_index(vertices,point,Vol)
            relativePoin = relativePlace_calculate(alpha,beta,gamma,delta,vertices_new)
            points_new.append(relativePoin)
        else:
            continue

fig = plt.figure()

ax = fig.add_subplot(121, projection='3d')
faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
ax.add_collection3d(mplot3d.art3d.Poly3DCollection(tetra_vertices, color='lightgrey'))

# 添加坐标点
ax.scatter([p[0] for p in points], [p[1] for p in points], [p[2] for p in points],
           color='red', s=100, alpha=1, depthshade=False)

# 设置坐标轴范围和标签,这里其实想写一个自动画范围的东西
# ax.set_xlim([-0.2, 1.2])
# ax.set_ylim([-0.2, 1.2])
# ax.set_zlim([-0.1, 0.3])
ax.auto_scale_xyz(scale, scale, scale)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


#绘制变形后散点分布
ay = fig.add_subplot(122, projection='3d')
# 可视化一些系列四面体
ay.add_collection3d(mplot3d.art3d.Poly3DCollection(tetra_vertices_new, color='lightgrey'))

# 添加坐标点
ay.scatter([p[0] for p in points_new], [p[1] for p in points_new], [p[2] for p in points_new],
           color='red', s=10, alpha=1, depthshade=False)

# 设置坐标轴范围和标签
# ay.set_xlim([-4, 4])
# ay.set_ylim([-4, 4])
# ay.set_zlim([-1, 4])
ay.auto_scale_xyz(scale_new, scale_new, scale_new)

ay.set_xlabel('X')
ay.set_ylabel('Y')
ay.set_zlabel('Z')

plt.show() # 显示图像















pass
# # 2、Your_mesh中的vector作为四面体
# for i in range(0,len(Your_mesh.Vectors)):
#   四面体 = Your_mesh.vector[i][0123]
#   new_四面体 = Your_mesh_new.vector[i][0123]

#   Vol = caculate(四面体)

#   # 3、判断点是否在该四面体内
#   for point in points:
#     if is_inpoint(point):
#       Index = PositionIndex(point，四面体，vol)# 计算参数
#       new_position = new_position(poin,四面体_new,Index)
#     else:
#       continue
# # 4、可视化出来四面体list
# show(四面体list，Points)
