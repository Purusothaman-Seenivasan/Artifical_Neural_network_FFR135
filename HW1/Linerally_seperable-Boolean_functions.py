import numpy as np
import random as random
from itertools import product
dimension = [1,2,3,4]
noOfTrails = 10**4
epacks= 20
eta=0.05
numberOfSeperable=0

input_dict={2:[],3:[],4:[], 5:[]}
target_dic= {2:[],3:[],4:[],5:[]}
#initialize input of the boolean function
def Set_input(dim):
  combinations = list(product([0, 1], repeat=dim))
  matrix = np.array(combinations)
  return matrix

def Set_target(dim):
  combinations = list(product([-1, 1], repeat=2**dim))
  matrix = np.array(combinations)
  return matrix

def Input_output(input, output):
  io_list=[]
  dim= len(input[1])
  for j in range(0,2**(2**dim)):
    for i in range(0,2**dim):
      temp=[]
      temp.append(input[i])
      temp.append(output[j][i])
      io_list.append(temp)
  return io_list


for n in dimension:
  input_dict[n] = Set_input(n)
  target_dic[n]= Set_target(n)
  numberOfSeperable=0
  #initialize perceptor
  #l = np.zeros((1, n))
  #weight=np.random.normal(1/((n)^int((1/2))),l)
  weight=np.random.normal(scale=1/np.sqrt(n),size=(1,n))
  thetha=0
  #train perceptor
  prev_bool=[]


    #initialize perceptor
        #l = np.zeros((1, n))
        #weight=np.random.normal(1/((n)^int((1/2))),l)
        #thetha = 0

  # print(n)
  # for t in range(noOfTrails):
  #   print(len)
  #   if len(prev_bool)== 2**(2**n):
  #     break
  for k in range(2**(2**n)):
    for e in range(0,epacks):
        target = target_dic[n][k]
        target_list = []
        function_output=[]
        for i in range(0,2**n):
            x= input_dict[n][i]
            localField= np.dot(weight,x.transpose())-thetha
            output=np.where(localField>0, 1,-1)
            weight += eta*( target[i] - output)*x
            thetha -= eta *(target[i] - output)
            target_list.append([target[i]])
            function_output.append(output)
        #if e==19:
          # print(target_list, 'output=', function_output)


        if target_list == function_output:
            if target_list not in prev_bool:
                numberOfSeperable+=1
                #print(target_list)
                prev_bool.append(target_list)
                break
  print('For', n, 'dimension number of linearlly seperable function is', numberOfSeperable)






