import numpy as np
import LAE_directs_solution as lds


# 求均差------------------------------------------------------------------------
def DivDiff(x,f,order):
#给出函数离散节点上的差商/均差(divided differences)
#从1阶最高算至order阶
    #数据点数
    n = x.size
    #初始化一个输出list
    val = []
    #当前差商为0阶
    currDD = f.copy()
    #当前差商长度
    l = n
    #开始循环，直到覆盖order阶
    for i in range(1,order+1):
        #自变量差
        diffx = x[i:n]-x[0:n-i]
        #函数差，逐阶递推
        df = currDD[1:l]-currDD[0:l-1]
        #更新当前差商
        l = l-1
        currDD = np.zeros((l,))
        currDD = df/diffx
        val.append(currDD)
    return val


# Hermint插值-----------------------------------------------------------------------
def cubicHermite(xx, x, f, df):

    a0 = (1+2*(xx-x[0])/(x[1]-x[0]))*((xx-x[1])/(x[0]-x[1]))**2
    a1 = (1+2*(xx-x[1])/(x[0]-x[1]))*((xx-x[0])/(x[1]-x[0]))**2
    b0 = (xx-x[0])*((xx-x[1])/(x[0]-x[1]))**2
    b1 = (xx-x[1])*((xx-x[0])/(x[1]-x[0]))**2

    val = f[0]*a0+f[1]*a1+df[0]*b0+df[1]*b1
    return val


# 三次样条插值
# 待插值点，插值节点，插值节点函数值，边界条件类型， 边界条件可选参数
def cubicSpline(x_intp, x, f, method, *arg):
    # 提取边界条件
    if method == 1:
        df0, dfn = arg[0], arg[1]
    elif method == 2:
        d2f0, d2fn = arg[0], arg[1]
    else:
        print('请输入正确的边界条件方法')
        return None

    # 统一n，注意，有n+1个插值节点，但是下表从0到n，这里的n与下标统一
    n = len(x)-1

    # 计算所有的二阶均差
    mean_df = DivDiff(x, f, 2)
    mean_df2 = mean_df[1]

    #计算h，mu， lamda
    h = x[1:n+1] - x[0: n]
    mu = h.copy()
    for j in range(1, n):
        mu[j] = h[j-1]/(h[j-1]+h[j])
    lamda = -1*mu + 1
    
    # 计算d向量，构造方程组
    if method == 1:
        # 边界条件1
        d = np.zeros(n+1,)
        d[0] =  6 * ((f[1]-f[0])/(x[1]-x[0])**2 - df0/(x[1]-x[0]))
        d[n] =  6 * (dfn/(x[n]-x[n-1]) - (f[n]-f[n-1])/(x[n]-x[n-1])**2)
        for i in range(n-1):
            d[i+1] = 6 * mean_df2[i]

        A = np.zeros((n+1, n+3))
        A[0, 1], A[0, 2] = [2, 1]
        A[n, n], A[n, n+1] = [1, 2]
        for i in range(n-1):
            A[i+1, i+1] = mu[i+1]
            A[i+1, i+2] = 2
            A[i+1, i+3] = lamda[i+1]
        A = A[:, 1:n+2]

        # 求解M(numpy函数)
        # M = np.linalg.solve(A, d)
        # 自己写的三对角追赶法
        M = lds.solveTridiagA(A, d)


    elif method == 2:
        # 边界条件2
        d = np.zeros(n-1,)
        for i in range(n-1):
            d[i] = 6 * mean_df2[i]
        d[0] -= mu[1]*d2f0
        d[n-2] -= lamda[n-1]*d2fn

        A = np.zeros((n-1, n+1))
        for i in range(n-1):
            A[i, i] = mu[i+1]
            A[i, i+1] = 2
            A[i, i+2] = lamda[i+1]
        A = A[:, 1:n]

        # 求解M(numpy函数)
        # M1 = np.linalg.solve(A, d)
        # 自己写的三对角追赶法
        M1 = lds.solveTridiagA(A, d)
        M = np.concatenate((np.array([d2f0]), M1, np.array([d2fn])))

    j = np.searchsorted(x, x_intp)

    if x_intp == x[j]:
        return f[j]
    else:
        j -= 1
        s1 = M[j]/6/h[j]*(x[j+1]-x_intp)**3
        s2 = M[j+1]/6/h[j]*(x_intp-x[j])**3
        s3 = (f[j]-M[j]*h[j]**2/6) * (x[j+1]-x_intp)/h[j]
        s4 = (f[j+1]-M[j+1]*h[j]**2/6) * (x_intp-x[j])/h[j]
        return s1 + s2 + s3 + s4