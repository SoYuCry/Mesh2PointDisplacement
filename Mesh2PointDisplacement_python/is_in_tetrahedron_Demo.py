# newbing代码测试，散射体是否在四面体中
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Define the vertices of the tetrahedron
vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Create the Delaunay triangulation
tetra = Delaunay(vertices)

# Define the point to test
point = np.array([0, 0, 0])

# Find the simplex (tetrahedron) containing the point
simplex = tetra.find_simplex(point)

# Check if the point is inside the tetrahedron
if simplex != -1:
    print("The point is inside the tetrahedron.")
else:
    print("The point is outside the tetrahedron.")

# Plot the tetrahedron and the point
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
ax.add_collection3d(Poly3DCollection(vertices[faces], alpha=0.25, facecolor='lightskyblue'))
ax.scatter([point[0]], [point[1]], [point[2]], color='red', s=10, alpha=1, depthshade=False)

plt.show()

pass