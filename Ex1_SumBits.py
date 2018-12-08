# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 17:33:58 2018

@author: Pedro Campelo
"""

"""Exercise 3.8 — An algorithm processes some of the bits of the binary
representations of numbers from 1 to 2n − 1, where n is the number
of bits of each number. In particular, the algorithm processes the least
significant bits of each number (from right to left), until it finds a bit
set to 1. Given n, determine, using sums, the total number of bits that
the algorithm processes. For example, for n = 4 the algorithm processes
the 26 shaded bits in Figure 3.11."""

"""
"3.11) Um algorítimo processa alguns de uma representação binária de 1 até (2^(n) − 1), onde n é o número de bits de cada número. Em particular, o algorítimo processa pelo menos os bits significantes de cada número (da direita para a esquerda), até que ache o número 1. Dado n, determine, usando soma, o número de bits que o algorismo processa. Por exemplo, para n=4, o número total de bits é 26."

8.4) Resolve o problema 3.8 usando uma abordagem recursiva e desenhe o a figura original do problema 3.11.

A questão acima é a questão 8.4 do livro "Introduction to Recursive Programming" de Manuel Rubio Sánches.
"""



def soma_bits(n):
    if (n==1):
        return 1
    else:
        return (2*soma_bits(n-1)+n)
    


import itertools
import numpy as np


#matriz(2^n - 1 linhas e n colunas) que retorna a permutação de n  
#Lembrando que é preciso eliminar a primeira linha (0,0,..n)

def bin_perm(n): 
    lista=0
    lista=np.array(list(itertools.product([0, 1], repeat=n)))
    lista=np.delete(lista, (0), axis=0)
    return lista



import plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode

import plotly.graph_objs as go
import plotly.figure_factory as ff
plotly.offline.init_notebook_mode(connected=True)




table_data = [["Coluna 1", "Coluna 2","Coluna 3", "Coluna 4"],
              [0, 0, 0, 1],
               [0, 0, 1, 0],
               [0, 0, 1, 1],
               [0, 1, 0, 0],
               [0, 1, 0, 1],
               [0, 1, 1, 0],
               [0, 1, 1, 1],
               [1, 0, 0, 0],
               [1, 0, 0, 1],
               [1, 0, 1, 0],
               [1, 0, 1, 1],
               [1, 1, 0, 0],
               [1, 1, 0, 1],
               [1, 1, 1, 0],
               [1, 1, 1, 1]]

figure = ff.create_table(table_data)
plotly.offline.plot(figure)





"""py.iplot(data, filename = 'basic_table')


table_data = [['Team', 'Wins', 'Losses', 'Ties'],
              ['Montréal<br>Canadiens', 18, 4, 0],
              ['Dallas Stars', 18, 5, 0],
              ['NY Rangers', 16, 5, 0], 
              ['Boston<br>Bruins', 13, 8, 0],
              ['Chicago<br>Blackhawks', 13, 8, 0],
              ['LA Kings', 13, 8, 0],
              ['Ottawa<br>Senators', 12, 5, 0]]

figure = ff.create_table(table_data, height_constant=60)
py.iplot(figure, filename='subplot_table')
plotly.offline.plot(figure)"""








"""traco = go.Table(
        header=dict(values=["n=4","n=4","n=4","n=4"])
        cells=dict(values=[[0, 0, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 1, 1],
                           [0, 1, 0, 0],
                           [0, 1, 0, 1],
                           [0, 1, 1, 0],
                           [0, 1, 1, 1],
                           [1, 0, 0, 0],
                           [1, 0, 0, 1],
                           [1, 0, 1, 0],
                           [1, 0, 1, 1],
                           [1, 1, 0, 0],
                           [1, 1, 0, 1],
                           [1, 1, 1, 0],
                           [1, 1, 1, 1]]))


data = [trace]
plotly.offline.plot(trace)"""




"""
#Alguns testes antes

#tentando definir a função
def binary_permutations (n):
    for i in range(1<n):
        s=bin(i)[2:]
        s='0'*(n-len(s))+s
        print (map(int,list(s)))
binary_permutations (3)



    n=4
    lst_aux = list(itertools.product([0, 1], repeat=n))
    del(lst_aux[0])
    lst = [list(i) for i in lst_aux]
    print(len(lst))
     strn_lista= str(lst)


#lista que retorna a permutação binária de n
     
def bin_perm_aux (n):
    lista2 =[]
    lista2.append(list(itertools.product([0, 1], repeat=n)))
    lista2[0].pop(0)    
    return lista2




import itertools
import numpy as np


#matriz(2^n - 1 linhas e n colunas) que retorna a permutação de n  
#Lembrando que é preciso eliminar a primeira linha (0,0,..n)

def bin_perm(n): 
    lista=0
    lista=np.array(list(itertools.product([0, 1], repeat=n)))
    lista=np.delete(lista, (0), axis=0)
    return lista





#aqui sao formas diferentes que tentei chamar a soma. Essa é primeira e principal
#1)
    
def sum_bits(n):
     lista=bin_perm(n)
      aux=[]
     if (n==1):
         return 1
     else:
     for p in range (1,n+1):
         for i in range(0,linhas+1):
             while (lista[i][n-p]==1):
                 return p
         aux.append(p)
         return aux    

#1.1) Variação da primeira
             
def sum_bits(n):
     lista=bin_perm(n)
     aux=[]
     for p in range (1,n+1):
         for i in range(0,linhas+1):
             while (lista[i][n-p]==1):
                 return p
         aux.append(p)
         return aux


#2) Compilação com a sua 
 def sum_bits(n):
     lista=bin_perm(n)
     aux=0
     for i in range(0,linhas+1): 
         num1=False
         for p in range (1,n+1):
             if num1==False:
                 if (lista[i][n-p]==1):
                     return p
                     aux += p+1
                     num_um=True
         return aux
  
    
#3) Essa é as sua  
def sumbits(lista):
    soma=0
    for i in range(linhas+1):
        num_um=False
        for p in range (1,n+1):
            if num_um==False:
                if lista[i][p]==1:
                    soma += p+1
                    num_um=True
    return soma


#4) Outra tentativa          
            
 def sum_bits(n):
     lista=bin_perm(n)
     if (n==1):
         return 1
     else:
         aux=[]
         for i in range(0,linhas):
             for p in range (1,n):
                 if (lista[i][n-p]==1):
                    aux=np.array([p])
                    aux.append([p])
                    return sum(aux)


#5) Este são testes, aonde eu testava cada linha da matriz
# E testava se o segundo loop de retornar o numero de quadrados funcionava.
#Funcionava, porém na compliação com os dois loops nao roda a soma nem retorna uma lista.



y=bin_perm(3)

def teste(n):
     if (n==1):
         return 1
     else:
         aux=[]
         for p in range (1,n+1):
             while (y[3][n-p]==1):
                 return p



                 aux=lista2[i][-p]
                 aux.append(lista2[])
                 return sum(aux)

    for i in range (0,3)
              
"""


#EXERCICIO FELIPE

import numpy as np
import pylab
from random import randint

def pontoMedio(ponto1, ponto2):
    return [(ponto1[0]+ponto2[0])/2, (ponto1[1]+ponto2[1])/2]


ponto_atual = [0,0]

vertice_1 = [0,0]
vertice_2 = [1,0]
vertice_3 = [.5,np.sqrt(3)/2]


for _ in range(5000):

    val = randint(0,2)
    if val == 0:
        ponto_atual = pontoMedio(ponto_atual, vertice_1)
    if val == 1:
        ponto_atual = pontoMedio(ponto_atual, vertice_2)
    if val == 2:
        ponto_atual = pontoMedio(ponto_atual, vertice_3)

    pylab.plot(ponto_atual[0], ponto_atual[1], 'm.', markersize=2)    

pylab.show()



"OUTRO JEITO"

import turtle

# function to have the turtle draw a triangle, the basic unit of our fractal
def draw_triangle(vertices,color,my_turtle):
    my_turtle.fillcolor(color)
    my_turtle.up()
    my_turtle.goto(vertices[0][0],vertices[0][1])
    my_turtle.down()
    my_turtle.begin_fill()
    my_turtle.goto(vertices[1][0],vertices[1][1])
    my_turtle.goto(vertices[2][0],vertices[2][1])
    my_turtle.goto(vertices[0][0],vertices[0][1])
    my_turtle.end_fill()
    
# the same midpoint function we wrote for the chaos game
def midpoint(point1, point2):
    return [(point1[0] + point2[0])/2, (point1[1] + point2[1])/2]
    
# recursive function that draws the different "levels" of the fractal
def draw_fractal(vertices,level,my_turtle):
    # the different colors we'll use to draw the fractals
    # in RGB format
    colors = [(0,150,189),(4,150,116),(216,95,30),(193,33,57),(129,41,199),
                (102,205,135),(51,187,204)]
    draw_triangle(vertices,colors[level],my_turtle)
    # call function recursively to draw all levels of fractal
    if level > 0:
        # draw first segment of fractal
        # the vertices being passed in are the bottom corner of the first
        # section, the bottom corner of the second section, and the bottom
        # corner of the third secion.
        draw_fractal([vertices[0],
                      midpoint(vertices[0], vertices[1]),
                      midpoint(vertices[0], vertices[2])],
                      level - 1, my_turtle)
        draw_fractal([vertices[1],
                      midpoint(vertices[0], vertices[1]),
                      midpoint(vertices[1], vertices[2])],
                      level - 1, my_turtle)
        draw_fractal([vertices[2],
                      midpoint(vertices[2], vertices[1]),
                      midpoint(vertices[0], vertices[2])],
                      level - 1, my_turtle)

my_turtle = turtle.Turtle()
my_turtle.shape('turtle')
screen = turtle.Screen()
screen.colormode(255) # to use the RGB codes for the colors
vertices = [[-200, -100], [0, 200], [200, -100]]
level = 4 # how many recursions deep do we want to draw the fractal
draw_fractal(vertices, level, my_turtle)
screen.exitonclick()

  
