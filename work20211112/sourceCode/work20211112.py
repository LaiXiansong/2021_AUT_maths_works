import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as li
from scipy import interpolate

# own lib
import my_interpolation as itp

# 计算实习题第三题的数据
x = np.array([0.9,1.3,1.9,2.1,2.6,3.0,3.9,4.4,4.7,5,6,7,8,9.2,10.5,11.3,11.6,12,12.6,13,13.3])
f = np.array([1.3,1.5,1.85,2.1,2.6,2.7,2.4,2.15,2.05,2.1,2.25,2.3,2.25,1.95,1.4,0.9,0.7,0.6,0.5,0.4,0.25])
n = len(x)

# df0 = (f[1]-f[0])/(x[1]-x[0])
# df1 = (f[2]-f[1])/(x[2]-x[1])
# dfn = (f[n-1]-f[n-2])/(x[n-1]-x[n-2])
# dfn1 = (f[n-2]-f[n-3])/(x[n-2]-x[n-3])
# 计算的三阶均差
mean_df1 = itp.DivDiff(x, f, 1)[0]
mean_df2 = itp.DivDiff(x, f, 2)[1]
df0, dfn = mean_df1[0], mean_df1[n-2]
d2f0, d2fn = mean_df2[0], mean_df2[n-3]

xx = np.arange(0.9, 13.3, 0.01)
yy = np.zeros(len(xx))
for i in range(len(xx)):
    yy[i] = itp.cubicSpline(xx[i], x, f, 1, df0, dfn)
plt.subplot(3,1,1)
plt.plot(xx, yy)
plt.scatter(x, f)
plt.title('边界条件1')

# 边界条件2
for i in range(len(xx)):
    yy[i] = itp.cubicSpline(xx[i], x, f, 2, d2f0, d2fn)
plt.subplot(3,1,2)
plt.plot(xx, yy)
plt.scatter(x, f)

plt.title('边界条件2')

# scipy 函数
interp_f = interpolate.interp1d(x, f, 'cubic')
yy = interp_f(xx)
plt.subplot(3,1,3)
plt.plot(xx, yy)
plt.scatter(x, f)

plt.title('scipy')


# 中文字体显示设置
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

plt.show()