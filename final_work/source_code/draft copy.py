import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as li

# 中文字体显示设置
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


def f(y):
    lamda = [-0.1, -27.5]
    return np.array([y[0]*lamda[0], y[1]*lamda[1]])

def TrueSolv(x):
    lamda = [-0.1, -27.5]
    y = x.copy()
    y[0] = math.exp(lamda[0]*x[0])
    y[1] = math.exp(lamda[1]*x[1]) 

    return np.sqrt(1+2*x)

a, b = 0, 5
# 求解网络
n = 100
h = (b-a)/n
x0 = 0
x = x0+np.arange(0, n+1)*h

solver = np.empty((2, n+1))

solver[:, 0] = 1


for i in range(1, n+1):
    K1 = f(solver[:, i-1])
    K2 = f(solver[:, i-1]+0.5*h*K1)
    K3 = f(solver[:, i-1]+0.5*h*K2)
    K4 = f(solver[:, i-1]+h*K3)
    solver[:, i] = solver[:, i-1] + 1/6*h*(K1+2*K2+2*K3+K4)

# 精确

solve_acu = np.copy(x)
# solve_acu = TrueSolv(x)


print('误差:', abs(solve_acu[-1]-solver[:, -1]))

# plt.plot(x, solve_acu,color='r')
# plt.plot(x, solve_EE, color='r')
plt.plot(x, solver[0,:], color='g')
plt.plot(x, solver[1,:], color='g')
plt.show()

