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
output_weight = np.random.rand(40,40,4)
distance = np.zeros((40,40))
min_distance= 100000
(min_i,min_j) = (-1,-1)

intial_norm= np.full((input_size,1),1000000)
intial_winner=np.full((input_size,2), 1000000)
final_norm =np.full((input_size,1),1000000)
final_winner = np.full((input_size,2),1000000)

print('outpt weight', output_weight[0,1], output_weight[1,1])
print('input' ,input_data)


for i in range(input_size):
  # for j in range (40):
  intial_output= np.dot(output_weight,input_data[i])

  min_dis=100000
  winnerNeuron=(-1,-1)

  for j in range (40):
    for k in range (40):
      d= np.linalg.norm(input_data[i]-output_weight[j,k])
      # print('distance', d, i)
      if d < min_dis:
        winnerNeuron= (j,k)
        min_dis = d
  intial_winner[i,:] = winnerNeuron
  intial_norm[i] = min_dis
  print(intial_norm)

print(intial_winner)
      # if d<intial_norm[i]:
      #   intial_winner[i,:]= [j,k] #intial_output[j,k]
      #   intial_norm[i]=d
      #   if i<2:
      #     print("int winner",intial_winner[:4,:])
input_dictionary={"Versicolor":[],"Sestosa":[], 'Virginica':[] }
for i in range(input_size):
  if i<50:
    input_dictionary['Versicolor'].append(i)
  elif i>49 and i <100:
    input_dictionary['Sestosa'].append(i)
  elif i>99:
    input_dictionary['Virginica'].append(i)
plotnumber=0

colors = ['red', 'blue', 'green']
# plt.figure(figsize=(10,10))


def h(d,sigma):

  neigbour = np.exp((d**2)/(-2*sigma**2))
  return neigbour

for epoch in range(5):
  sigma = sigma0 if epoch==0 else sigma*np.exp(-0.05*epoch)
  eta = eta0 if epoch==0 else eta*np.exp(-0.01*epoch)
  for n in range(input_size):
    output=np.dot(output_weight,input_data[n])
    # for i in range(40):
    #   for j in range(40):
    #     d=np.linalg.norm(input_data[n]-output_weight[i,j])
    #     if d< final_norm[n]:
    #       final_winner[n]= output[i,j]
    min_dis=100000
    winnerNeuron=(-1,-1)
    for i in range (40):
      for j in range (40):
        d= np.linalg.norm(input_data[n]-output_weight[i,j])
        # print('distance', d, i)
        if d < min_dis:
          winnerNeuron= (i,j)
          min_dis = d
    final_winner[n,:] = winnerNeuron
    final_norm[n] = min_dis

    for i in range(40):
      for j in range(40):
        (win_x, win_y) = final_winner[n]
        # euclid_d= np.linalg.norm(output[win_x,win_y] -output[i,j])
        euclid_d=0
        euclid_d = np.linalg.norm(np.array([i,j])-final_winner[n])
        output_weight[i][j] = output_weight[i][j] + eta*h(euclid_d,sigma)*(input_data[n]-output_weight[i][j])
      # print("h", h(euclid_d,sigma))
print(euclid_d)
print(final_winner)


legend_labels = ['Versicolor', 'Sestosa', 'Virginica']
fig,(ax1,ax2) = plt.subplots(1,2)
plt.xlim(0,40)
plt.ylim(0,40)
for lab in range(3):
  indices= [] #[int(x) for x in input_labels if int(x)==labels]
  for x in range(150):
    if int(float(input_labels[x][0])) == lab:
      indices.append(x)
  for index in indices:
    plotnumber +=1
  #   if index <50:
  #     legends = 'Versicolor'
  #   elif i>49 and i <100:
  #     legends = "Sestosa"
  #   else:
  #     legends = "Virginica"
    ax1.scatter(intial_winner[index][0],intial_winner[index][1], c=colors[lab])
    ax2.scatter(final_winner[index][0],final_winner[index][1], c=colors[lab])

ax1.set_title('Initial classification before updating weights')
ax2.set_title('Final classification after 10 epochs')

ax1.legend(handles = colors, labels= legend_labels)
ax2.legend(handles = colors, labels= legend_labels)

plt.show()





# plt.xlim(0,40)
# plt.ylim(0,40)
# for labels in range(3):
#   indices= [] #[int(x) for x in input_labels if int(x)==labels]
#   for x in range(150):
#     if int(float(input_labels[x][0])) == labels:
#       indices.append(x)
#   for index in indices:
#     plotnumber +=1
#     ax2.plot.scatter(final_winner[index][0],final_winner[index][1], c=colors[labels])
# plt.show()