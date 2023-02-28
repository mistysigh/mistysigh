# source : 人工智数学基础 page31

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def Fun(x, y):
    return x - y + 2 * x * x + 2 * x * y + y * y


def PxFun(x, y):
    return 1 + 4 * x + 2 * y


def PyFun(x, y):
    return -1 + 2 * x + 2 * y


fig = plt.figure()
ax = Axes3D(fig)
X, Y = np.mgrid[-2:2:40j, -2:2:40j]
Z = Fun(X, Y)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# 梯度下降
step = 0.0008
x = 0
y = 0
tag_x = [x]
tag_y = [y]
tag_z = [Fun(x, y)]
new_x = x
new_y = y
Over = False
while not Over:
    new_x -= step * PxFun(x, y)
    new_y -= step * PyFun(x, y)
    if Fun(x, y) - Fun(new_x, new_y) < 7e-9:
        Over = True
    x = new_x
    y = new_y
    tag_x.append(x)
    tag_y.append(y)
    tag_z.append(Fun(x, y))

ax.plot(tag_x, tag_y, tag_z, 'r')
plt.title('(x,y)~(' + str(x) + "," + str(y) + ')')
plt.show()