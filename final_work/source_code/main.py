import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import scipy.linalg as li

# 中文字体显示设置
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


def f(input):
    # input[] = t, x, vx, y, vy
    # t = input[0]
    x = input[0]
    vx = input[1]
    y = input[2]
    vy = input[3]
    return np.array([vx, -9.8*x/(6.371*(x**2 + y**2)**(1.5)), vy, -9.8*y/(6.371*(x**2 + y**2)**(1.5))])


a, b = 0, 10
# 求解网络
n = 1000
h = (b-a)/n

# 初始化时间变量（1unit = 1000s）
t0 = 0
t = t0+np.arange(0, n+1)*h

euler_solver = np.empty((4, n+1))
rk4_solver = np.empty((4, n+1))


# 变量初始化
# 第一宇宙速度为7.9km/s， 归一化后约为1.24
euler_solver[:, 0] = np.transpose([1, 0, 0, 1.5]) # [x0, vx0, y0, vy0]
rk4_solver[:, 0] = np.transpose([1, 0, 0, 1.5]) # [x0, vx0, y0, vy0]



for i in range(1, n+1):
    K1 = f(rk4_solver[:, i-1])
    K2 = f(rk4_solver[:, i-1]+0.5*h*K1)
    K3 = f(rk4_solver[:, i-1]+0.5*h*K2)
    K4 = f(rk4_solver[:, i-1]+h*K3)
    euler_solver[:, i] = euler_solver[:, i-1] + h*f(euler_solver[:, i-1])
    rk4_solver[:, i] = rk4_solver[:, i-1] + 1/6*h*(K1+2*K2+2*K3+K4)

# 画中心圆用到的参数
theta = linspace(0, 2*np.pi, num=100)
r = 0.8
rx = r * np.sin(theta)
ry = r * np.cos(theta)



plt.plot(rx, ry, 'g', label='圆周轨迹')
plt.plot(euler_solver[0,:], euler_solver[2,:], 'b', label='显式Euler求得轨迹')
plt.plot(rk4_solver[0,:], rk4_solver[2,:], 'r', label='RK4求得轨迹')
plt.legend(['圆周轨迹', '显式Euler求得轨迹', 'RK4求得轨迹'])
plt.axis('scaled')
plt.show()

print(rk4_solver[:,-1])