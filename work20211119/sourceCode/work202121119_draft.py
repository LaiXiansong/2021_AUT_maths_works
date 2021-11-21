import numpy as np
import matplotlib.pyplot as plt
# my lib
import my_approximation as fit


# 中文字体显示设置
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
fx = np.array([0.97, 0.83, 0.65, 0.54, 0.46, 0.36, 0.29, 0.25, 0.21, 0.17])

order = 6

# 调用施加扰动的拟合函数
a, err = fit.myLeastSqDist(x, fx, order)

xx = np.arange(0, 1, 0.01)
yy = np.arange(0, 1, 0.01)
for i in range(len(xx)):
    yy[i] = fit.fitFunction(xx[i],a)
plt.plot(xx, yy)
plt.scatter(x,fx)
plt.title('%d阶线性无关多项式拟合(扰动)' % (order))
plt.show()

print(order, "阶error：", err)