import numpy as np
import math
import nl_eq_iteration as nleqi
from itertools import combinations as cb


###################################################################################
# assuming that we don't know the value of m
# so we can use the 2rd method for multiple roots
def MrFx(x, xi):
    return nleqi.omigaPolynomial(x, xi, 0)


def dMrFx(x, xi):
    return nleqi.omigaPolynomial(x, xi, 1)


def d2MrFx(x, xi):
    return nleqi.omigaPolynomial(x, xi, 2)


def mrNewtonIterF(x, xi: np.ndarray):
    return x - (MrFx(x, xi)*dMrFx(x, xi)) / ((dMrFx(x, xi)**2)-(MrFx(x, xi)*d2MrFx(x, xi)))


xi = np.array([[1, 2, 3, 4]])
root_nums = xi.shape[1]
x_0 = 1
x_solve_mn = []
for i in range(root_nums):
    # only one root left
    if xi.shape[1] == 1:
        x_solve_mn.append(xi[0, 0])
        break
    else:
        x_star = nleqi.mrNewtonFP(x_0, 10**(-3), mrNewtonIterF, xi)
        x_solve_mn.append(x_star)
        temp = np. zeros((1, xi.shape[1]-1))
        k = 0
        for j in range(xi.shape[1]):
            if round(xi[0, j]) != round(x_star):
                temp[0, k] = xi[0, j]
                k += 1
            if k == temp.shape[1]:
                break
        xi = temp.copy()
print('iteration result of fixed newton method: ', x_solve_mn)


