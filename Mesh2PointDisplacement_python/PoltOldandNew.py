
# 可以在main.py DEBUG的时候，手动把Points和Points_new导出成.npy文件，用这个单独显示新旧散射点位置
import numpy as np
import matplotlib.pyplot as plt

# a = [2,3]
# a=np.array(a)
# np.save('a.npy',a) # 保存为.npy格式

# a=np.load('a.npy')
# a=a.tolist()
points = np.load('points.npy')
points_new = np.load('points_new.npy')

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')

ay = fig.add_subplot(122, projection='3d')



ax.scatter([p[0] for p in points], [p[1] for p in points], [p[2] for p in points],
           color='red', s=10, alpha=1, depthshade=False)
ay.scatter([p[0] for p in points_new], [p[1] for p in points_new], [p[2] for p in points_new],
           color='red', s=10, alpha=1, depthshade=False)
           
plt.show()
pass