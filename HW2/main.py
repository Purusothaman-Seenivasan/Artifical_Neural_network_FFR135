import csv
import random
import numpy as np
import matplotlib.pyplot as plt

#extracting data from csv file
#extracting data from csv file
intialdata = []  #store data as it is, as in csv
data = []  #stores normalized vectors
eta = 0.01
epacks = 400
number_neruons = 50
batch_size = 50
xmean = 0
ymean = 0
validation_data = []
xplus = 0
yplus = 0
n_validation = []
(x_mean_validation_plus, y_mean_validation_plus) = (0, 0)
x_mean_validation = 0
y_mean_validation = 0
train_target = []
valid_target = []
with open('training_set.csv', 'r') as train_file:
  csv_reader = csv.reader(train_file)
  for line in csv_reader:
    x = float(line[0])
    y = float(line[1])
    xplus += x
    yplus += y
    vector = [x, y]
    intialdata.append(vector)
    train_target.append(float(line[2]))

print(len(intialdata))
train_mean = np.mean(intialdata, axis=0)
print('mean', train_mean)
train_centered = intialdata - train_mean
print('meant', train_centered)
train_variance = np.var(train_centered, axis=0)
print('variance', train_variance)
norm_data = train_centered / np.sqrt(train_variance)
print('norm', norm_data)
for n in range(len(norm_data)):
  data.append([norm_data[n][0], norm_data[n][1], train_target[n]])
variance = np.var(norm_data, axis=0)
print('variance', variance)
# print(data[:20])
with open('validation_set.csv', 'r') as validation_file:
  csv_reader1 = csv.reader(validation_file)
  for line in csv_reader1:
    x = float(line[0])
    y = float(line[1])
    x_mean_validation_plus += x
    y_mean_validation_plus += y

    vector_validation = [x, y]
    validation_data.append(vector_validation)
    valid_target.append(float(line[2]))

validation_mean = np.mean(validation_data, axis=0)
validation_centered = validation_data - validation_mean
validation_variance = np.var(validation_centered, axis=0)
norm_validation = validation_centered / np.sqrt(validation_variance)

for o in range(len(norm_validation)):
  n_validation.append(
      [norm_validation[o][0], norm_validation[o][1], valid_target[o]])
valid_variance = np.var(norm_validation, axis=0)
print('nvariance', valid_variance)

# print((n_validation[:10]))


#----------------------------------------------------------------------------------
#intilize weight
#print(training_data)
def Intialize_weight(n):
  weight = np.random.normal(1 / np.sqrt(n), size=(1, n))
  return weight


db = np.zeros(((batch_size), 1))


def Calculate_db(x):
  dx = 1 - (np.tanh(x))**2
  return dx


hidden_2_output_weight = 0
input_2_hidden_weight = np.zeros((number_neruons, 2))
hidden_theta = np.zeros((number_neruons, 1))
#hidden_2_output_weight = np.zeros((1, number_neruons))

for n in range((number_neruons)):
  weight = Intialize_weight(2)
  input_2_hidden_weight[n] = weight
  hidden_theta[n] = 0  #random.randint(0, 1)
hidden_2_output_weight = Intialize_weight(number_neruons)
output_theta = 0
# print('w1', input_2_hidden_weight)
# print('w2', hidden_2_output_weight)
# print('htheta', hidden_theta)
# print('otheta', output_theta)

output_list = []
output_list_trail = []
#print(input_2_hidden_weight, hidden_theta)
local_field_hv = np.zeros((batch_size, number_neruons))
states_hn = np.zeros((batch_size, number_neruons))
#outputparameters
local_field_output = np.zeros((batch_size, 1))
output = np.zeros((batch_size, 1))
#-
local_field_hv_validation = np.zeros((batch_size, number_neruons))
states_hn_validation = np.zeros((batch_size, number_neruons))

#delta_output_error=[]
delta_output_error = np.zeros((batch_size, 1))
delta_hidden_layer = np.zeros((1, number_neruons))
####
local_field_hv_validation = np.zeros((len(n_validation), number_neruons))
states_hn_validation = np.zeros((len(n_validation), number_neruons))
# print('delta_output_error[i]', delta_output_error.shape)
# print('training_data[i][2]', data[0][2])
# print('local_field_hv', local_field_hv.shape)
#----------------------------------------------------------------------------
for r in range(epacks):
  ddata = data.copy()

  for d in range(int(len(data) / batch_size)):
    training_data = ddata[:batch_size]
    ddata = ddata[batch_size:]
    # training_data= random.shuffle(training_data)
    for i in range((len(training_data))):
      x = np.array(training_data[i][:-1])

      for v in range(number_neruons):
        local_field_hv[i][v] = np.dot(input_2_hidden_weight[v].reshape(1, 2),
                                      x.reshape(2, 1)) - hidden_theta[v]
        states_hn[i][v] = np.tanh(local_field_hv[i][v])

      local_field_output[i] = (np.dot(hidden_2_output_weight,
                                      states_hn[i].transpose())) - output_theta
      # if i < 5 and d == 1:
      #   print('local field',local_field_output)
      #   print('dot product error', hidden_2_output_weight, states_hn[i])
      #   print(np.dot(hidden_2_output_weight, states_hn[i].transpose())- output_theta)
      output[i] = np.tanh((local_field_output[i]))

      # print('output',output.shape)
      #output_list_trail.append(local_field_output)
      #-----------updating output weight and theta-----------------------
      delta_output_error[i] = (training_data[i][2]-output[i]) * (
          Calculate_db(local_field_output[i]))
      output_list.append(output)
    # print('local field hn', local_field_hv[:10])
    # print('after tan',states_hn[:10])
    # print('local field outptu', local_field_output[:10])
    # print('output', output[:10])

    for f in range(batch_size):
      hidden_2_output_weight += (delta_output_error[f] * states_hn[f]) * eta
      output_theta -= eta * delta_output_error[f]

      #updating input weight and theta------------------------------
      # print('gdash', prime_hidden_state.transpose().shape)
      # print('delta output error', delta_output_error[0].reshape(1, 1).shape)
      # print('hidden 2 output weight', hidden_2_output_weight.shape)
      # print("delta hidden", delta_hidden_layer.shape)
      # print('hidden theta', hidden_theta.transpose().shape)
#---------------------------check from here-------------------
    prime_hidden_state = np.zeros((number_neruons, 1))
    for g in range(batch_size):
      for y in range((number_neruons)):
        prime_hidden_state[y] = Calculate_db(local_field_hv[g][y])
      delta_hidden_layer = (delta_output_error[g]) * (
          hidden_2_output_weight) * prime_hidden_state.transpose()
      hidden_theta -= delta_hidden_layer.transpose() * eta

      input_2_hidden_weight += eta * (delta_hidden_layer.transpose() *
                                      training_data[g][:-1])
  # print('batch', d,'vj', states_hn)
  # print('local field output', output_list_trail[10:20])

  print('epack', r)
  # print(output[:10], training_data[:10][2])
  # print('hidden weight', hidden_2_output_weight)
  # print('hidden theta', hidden_theta)
  # print('input weight', input_2_hidden_weight)
  # print('output theta', output_theta)
  # print('-----------------------------')

  #--------------------------validation----------------------------
  output_validation_list = []

  def sgn(x):
    if x < 0:
      return -1
    else:
      return 1

  # print('input_2_hidden_weight', input_2_hidden_weight)
  # print('hidden_2_output_weight', hidden_2_output_weight)
  # print('hidden theta', hidden_theta)
  # print('output theta', output_theta)
  # print('v', states_hn.shape)
  # print('-----------------------------')
  # print('shape', delta_hidden_layer)
  local_field_output_validation = 0
  for j in range(len(n_validation)):
    x = np.array(n_validation[j][:-1])
    for v in range(number_neruons):
      local_field_hv_validation[j][v] = np.dot(input_2_hidden_weight[v],
                                               x.transpose()) - hidden_theta[v]
      states_hn_validation[j][v] = np.tanh(local_field_hv_validation[j][v])

    local_field_output_validation = (np.dot(
        hidden_2_output_weight,
        states_hn_validation[j].transpose())) - output_theta
    output_validation = np.tanh((local_field_output_validation))
    output_validation_list.append(output_validation)
  print(np.array(output_validation_list).shape)

  #print('output-validation first 10',output_validation_list[:10])
  z = 0
  count = 0
  # print(len(n_validation))
  for j in range(len(n_validation)):
    z += abs((sgn(output_validation_list[j])) - n_validation[j][2])
    if output_validation_list[j] < 1:
      count += 1
    if j <10:
      print('z', output_validation_list[j], sgn(output_validation_list[j]),
            n_validation[j][2])

  classificaton_error = (1 / (2 * (len(n_validation)))) * z
  print('#less than1', count)
  print('error', classificaton_error)

with open('w1.csv', mode='w', newline='') as file:
  csv_writer = csv.writer(file, delimiter=',')
  csv_writer.writerows(input_2_hidden_weight)

with open('w2.csv', mode='w', newline='') as file:
  csv_writer = csv.writer(file, delimiter=',')
  csv_writer.writerows(hidden_2_output_weight.transpose())

with open('t1.csv', mode='w', newline='') as file:
  csv_writer = csv.writer(file, delimiter=',')
  csv_writer.writerows(hidden_theta)

with open('t2.csv', mode='w', newline='') as file:
  csv_writer = csv.writer(file, delimiter=',')
  csv_writer.writerows([output_theta])

# #------------ploting input--------
# x_blue, y_blue, _ = zip(*[point for point in n_validation if point[2] == 1])
# x_red, y_red, _ = zip(*[point for point in n_validation if point[2] == -1])

# # Create the scatter plot
# plt.scatter(x_blue, y_blue, color='blue', label='1')
# plt.scatter(x_red, y_red, color='red', label='-1')

# # Set labels and legend
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend(title='Third Column')

# # Show the plot
# plt.show()
#-------------rnd ploting-----------------------------
