import numpy as np
import random
import csv 
from matplotlib import pyplot as plt 

with open('iris-data.csv', 'r') as f:
    reader = csv.reader(f)
    input = list(reader)

with open('iris-labels.csv', 'r') as f:
  reader = csv.reader(f)
  input_labels = list(reader)

sigma0 = 10
eta0 =0.1
dsigma = 0.05
deta= 0.01

input = np.array(input)
input= input.astype(float)

max_values = input.max(axis=0)
input_data = input/max_values


input_size = len(input_data)
print(np.array(input_data).shape)
print(len(input_data))

intial_output= np.zeros((40,40))
output = np.zeros((40,40))
output_weight = np.random.randint(0,41,size=(40,40,4))
distance = np.zeros((40,40))
min_distance= 100000
(min_i,min_j) = (-1,-1)

intial_norm= np.full((input_size,1),1000000)
intial_winner=np.full((input_size,2), 1000000)
final_norm =np.full((input_size,1),1000000)
final_winner = np.full((input_size,2),1000000)

print('outpt weight', output_weight[0,1])
print('input' ,input_data[0])


for i in range(input_size):
  # for j in range (40):
  intial_output= np.dot(output_weight,input_data[i])
  for j in range (40):
    for k in range (40):
      d= np.linalg.norm(input_data[i]-output_weight[j,k])
      # print('distance', d, i)
      if d<intial_norm[i]:
        intial_winner[i,:]= [j,k] #intial_output[j,k]
        intial_norm[i]=d