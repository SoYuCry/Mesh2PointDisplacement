
import numpy as np
from scipy.spatial import Delaunay
from stl import mesh

# 查看顶点是否在列表里
def is_unique(newpoint,points):
    flag = False
    for point in points:
        if newpoint == point:
            flag = True
            break
    return flag
        


filename = 'NormalMesh.stl'
mesh = mesh.Mesh.from_file(filename)
scale = mesh.points.flatten()
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

points = np.hstack([x, y, z])
DT = Delaunay(points)   

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# extract the tetrahedral elements from the Delaunay triangulation
tetra_vertices = DT.points[DT.simplices]

# create Poly3DCollection object
mesh = Poly3DCollection(tetra_vertices, alpha=0.3)

# add the mesh to the plot
ax.add_collection3d(mesh)

# set the axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.auto_scale_xyz(scale, scale, scale)
# show the plot
plt.show()