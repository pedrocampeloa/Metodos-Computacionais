# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 22:27:55 2018

@author: Felipe
"""


def josephus(n,k):
    r = 0
    for i in range(1,n+1):
        r = (r+k)%i
    return 'Survivor: %d' %r
 
print(josephus(6, 1))
print(josephus(41, 3))


 
## i=1: k



def j(n, k):
	p, i, seq = list(range(n)), 0, []
	while p:
		i = (i+k-1) % len(p)
		seq.append(p.pop(i))
	return 'Prisoner killing order: %s.\nSurvivor: %i' % (', '.join(str(i) for i in seq[:-1]), seq[-1])
 
print(j(5, 2))
print(j(41, 3))

def kill(people, passes):
    #If there is one person left, no need to continue.
    if people ==1:
        return 0
    return ((kill(people-1,passes)+passes) % people )

people = int(input("Please enter how many people (N) :"))

passes = int(input("Please enter number of passes (M) :"))

winner = kill(people, passes)-1
if winner < 0:
    winner = people

print("Player %d wins!" %(winner))



def josephus(n, k):
 
    if (n == 1):
        return 1
    else:
     
     
          # The position returned by 
          # josephus(n - 1, k) is adjusted
          # because the recursive call
          # josephus(n - 1, k) considers
          # the original position 
          # k%n + 1 as position 1 
          return (josephus(n - 1, k) + k-1) % n + 1
      
 
# Driver Program to test above function
n=5
k=1
 
print("The chosen place is ", josephus(n, k))





def josephus(pessoas, passes):
        i = passes
        
        while len(pessoas) > 1:
            print(pessoas.pop(i)) # 'mata' a pessoa em i
            i = (i + passes) % len(pessoas)
        print('Sobrevivente: ', pessoas[0])
        
josephus([1,2,3,4,5,6],1)







