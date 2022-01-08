import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as li

# 中文字体显示设置
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


def f(x, y):
    # return y-2*x/y
    return 1/y

def TrueSolv(x):
    return np.sqrt(1+2*x)

a, b = 0, 10
# 求解网络
n = 100
h = (b-a)/n
x = a+np.arange(0, n+1)*h

# 显式Euler
solve_EE = np.zeros((n+1,))
# 初值条件
solve_EE[0] = 1

for i in range(1, n+1):
    solve_EE[i] = solve_EE[i-1] + h*f(x[i-1], solve_EE[i-1])

# 改进Euler
solve_ME = np.zeros((n+1,))
# 初值条件
solve_ME[0] = 1

temp = solve_ME.copy()

for i in range(1, n+1):
    temp[i] = temp[i-1] + h*f(x[i-1], temp[i-1])
    solve_ME[i] = solve_ME[i-1] + h/2*(f(x[i-1], solve_ME[i-1]) + f(x[i], temp[i]))

# RK4
solve_RK4 = np.zeros((n+1,))
solve_RK4[0] = TrueSolv(a)

for i in range(1, n+1):
    K1 = f(x[i-1], solve_RK4[i-1])
    K2 = f(x[i-1]+0.5*h, solve_RK4[i-1]+0.5*h*K1)
    K3 = f(x[i-1]+0.5*h, solve_RK4[i-1]+0.5*h*K2)
    K4 = f(x[i-1]+h, solve_RK4[i-1]+h*K3)
    solve_RK4[i] = solve_RK4[i-1] + 1/6*h*(K1+2*K2+2*K3+K4)

# 精确

solve_acu = np.copy(x)
solve_acu = TrueSolv(x)


print('误差:', abs(solve_acu[-1]-solve_RK4[-1]))

plt.plot(x, solve_acu,color='r')
# plt.plot(x, solve_EE, color='r')
plt.plot(x, solve_RK4, color='g')
plt.title('=100')
plt.show()

