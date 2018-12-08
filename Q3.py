# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 22:26:52 2018

@author: Felipe
"""

import math

def factorial(m,n):
    c=m
    if(m!=n):
        p=math.floor((n-m)//2)
        a=factorial(m,m+p)
        b=factorial(m+p+1,n)
        c=a*b # a função veio pra dentro da outra #
    return c

print(factorial(1,4))