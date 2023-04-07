from mpl_toolkits import mplot3d
from matplotlib import pyplot
from stl import mesh

# 本地文件
filename = 'NormalMesh.stl'
# 创建一个plot
figure = pyplot.figure()
axes = figure.add_subplot(projection='3d')
# 加载stl文件，把读取到的向量信息加载到plot
mesh = mesh.Mesh.from_file(filename)
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors, color='lightgrey'))
#axes.plot_surface(mesh.x,mesh.y,mesh.z)
# 自动缩放网格尺寸
scale = mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)
#是否显示坐标轴
pyplot.axis('off')
#这里可以调整模型的角度
axes.view_init(azim=0)
# 保存到本地文件
pyplot.savefig('NormalMesh.png')
