#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as li
import scipy.special._comb as C

# own lib
import nl_eq_iteration as nleqi

###################################################################################
# P116-1 (1)、(2)、(5)


def Fx(x):
    return x**3 + 2*x**2 + 10*x - 20


def dFx(x):
    return 3*x**2 + 4*x + 10


def newtonIter(x):
    return x - Fx(x) / dFx(x)


def iterFx1(x):
    return 20 / (x**2 + 2*x +10)


def iterFx2(x):
    return (20 - 2*x**2 - x**3)/10


x_old = 1
x_solve1 = nleqi.fixedPtIter(x_old, 10**(-9), iterFx1)
print('iteration result of eq1: ', x_solve1)
x_solve2 = nleqi.fixedPtIter(x_old, 10**(-9), iterFx2)
print('iteration result of eq2: ', x_solve2)
x_solve_newton = nleqi.fixedPtIter(x_old, 10**(-9), newtonIter)
print('iteration result of newton method: ', x_solve_newton)