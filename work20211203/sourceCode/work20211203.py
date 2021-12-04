import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as li

import integration as integ

# 中文字体显示设置
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

# 生成cotes系数表
m, n = 9, 9
table = np.zeros((m, n))
for i in range(1, m):
    for j in range(0, i+1):
        table[i, j] = integ.cotes(i, j)

# 积分参数
a, b = 1, 3
def func(x):
    # f = np.log(x)
    f = (10/x)**2 * np.sin(10/x)
    return f

# for n in range(1, 9):
#     x = np.linspace(a, b, n+1, endpoint=True)
#     fx = x.copy()
#     for i in range(len(x)):
#         fx[i] = func(x[i])
#     inteCotes = 0
#     for k in range(n+1):
#         inteCotes += table[n, k] * fx[k]
#     print(n,"阶积分结果：",inteCotes)


# for n in range(2, 5000):
#     x = np.linspace(a, b, n+1, endpoint=True)
#     fx = x.copy()
#     for i in range(len(x)):
#         fx[i] = func(x[i])

#     h = (b - a) / n
#     # 复合梯形求积公式
#     cs = func(a) + func(b)
#     for k in range(1, n):
#         cs += 2 * func(x[k])
#     cs *= h/2
#     print(n,"节点梯形积分结果：", cs)
#     if np.abs(cs + 1.42602475634) < 1e-4:
#         print("到达精度的步长为：", h)
#         break

#     # 复合Simpson求积公式
#     sinp = func(a) + func(b)
#     for k in range(1, n):
#         sinp += 4 * func((x[k-1] + x[k])/2) + 2 * func(x[k])
#     sinp += 4 * func((x[k-1] + x[k])/2)
#     sinp *= h/6
#     print(n,"节点复合Simpson积分结果：", sinp)
#     if np.abs(sinp + 1.42602475634) < 1e-4:
#         print("到达精度的步长为：", h)
#         break

# Chebyshev
def gaussCheb(n, a, b, func):
    t = np.zeros((n+1, ))
    A = np.ones((n+1,)) * np.pi/(n+1)
    for k in range(n+1):
        t[k] = np.cos((2*k+1)*np.pi/2/(n+1))
    I = 0
    for k in range(n+1):
        I += np.sqrt(1-t[k]**2) * func((a+b)/2 + (b-a)/2*t[k])
    return I * (b-a)/2*np.pi/(n+1)

def combGaussCheb(m, n, a, b, func):
    x = np.linspace(a, b, m+1, endpoint=True)
    I_plus = 0
    for k in range(m):
        I_plus += gaussCheb(n, x[k], x[k+1], func)
    return I_plus

# for m in range(10000):
#     a, b = 1, 3
#     n = 150
#     I = combGaussCheb(m, n, a, b, func)
#     print(m, "节点复合", n,"阶Chebyshev积分:" ,I)
#     if np.abs(I + 1.42602475634) < 1e-4:
#         break

# def change(n):
#     for m in range(1000):
#         a, b = 1, 3
#         I = combGaussCheb(m, n, a, b, func)
#         if np.abs(I + 1.42602475634) < 1e-4:
#             return m

# xx = np.arange(80, 300, 10)
# yy = xx.copy()
# for i in range(len(xx)):
#     yy[i] = change(xx[i])
# plt.plot(xx, yy)
# plt.xlabel("阶次")
# plt.ylabel("节点个数")
# plt.show()

def combTrap(n, a, b, func):
    h = (b - a) / n
    x = np.linspace(a, b, n+1, endpoint=True)
    cs = func(a) + func(b)
    for k in range(1, n):
        cs += 2 * func(x[k])
    return cs * h/2

def Romberg(p, a, b, func):
    # p: Romberg的层数
    T = np.zeros((p, p))
    for i in range(p):
        T[i, 0] = combTrap(2**i, a, b, func)
        for k in range(1, i+1):
            T[i, k] = (4**k * T[i, k-1] - T[i-1, k-1]) / (4**k - 1)
    return T

for p in range(1, 20):
    T = Romberg(p, a, b, func)
    if np.abs(T[p-1, p-1] + 1.42602475634) < 1e-4:
        print(p, "层达到精度要求，结果：", T[p-1, p-1])
        break

print(T)
