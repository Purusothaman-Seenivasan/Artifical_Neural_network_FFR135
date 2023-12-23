import numpy as np


x1 = np.array([ [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] ])
x2= np.array([ [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1] ])
x3 = np.array([ [ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1] ])
x4 = np.array([ [ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, 1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1] ])
x5= np.array([ [ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, 1, 1, 1, 1, 1, 1, -1],[ -1, 1, 1, 1, 1, 1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1] ])


def matrix_to_column(z1):
  new_matrix=[]
  for i in range(0, len(z1)):
    for j in range(0, len(z1[0])):
      new_matrix.append(z1[i][j])
  return (new_matrix)
x1_column = matrix_to_column(x1)
x2_column = matrix_to_column(x2)
x3_column = matrix_to_column(x3)
x4_column = matrix_to_column(x4)
x5_column = matrix_to_column(x5)

weight_variable={} #dict to store all the weights
for i in range (0,5):
  x= f'w{i}'
  weight_variable[x]=[]

def weight (dic): #fun to intialize weight matrix
  no_of_bits= len(x1_column)
  for i in range (0,no_of_bits):
    x=[]
    for j in range (0, no_of_bits):
      x.append(0)
    dic.append(x)
  dic=np.array(dic)
  return dic

for a in range (0,5):
  weight_variable[f'w{a}'] = weight(weight_variable[f'w{a}'])



def weight_assigning (w,x): #fun to assign weight with hebbs rule
  for m in range (0, len(x)):
    for n in range(0, len(x)):
      if m!=n:
        w[m][n]=x[m]*x[n]
  return w

#weight_variable['w0']= weight_assigning(weight_variable['w1'], x1_column)

for b in range(0,5):
  if b==0:
    weight_variable[f'w{b}'] = weight_assigning(weight_variable[f'w{b}'],x1_column)
  elif b==1:
    weight_variable[f'w{b}'] = weight_assigning(weight_variable[f'w{b}'],x2_column)
  elif b==2:
    weight_variable[f'w{b}'] = weight_assigning(weight_variable[f'w{b}'],x3_column)
  elif b==3:
    weight_variable[f'w{b}'] = weight_assigning(weight_variable[f'w{b}'],x4_column)
  elif b==4:
    weight_variable[f'w{b}'] = weight_assigning(weight_variable[f'w{b}'],x5_column)

weight_of_stored_pattern= 0
for c in range(0,5):
  weight_of_stored_pattern+= weight_variable[f'w{c}']
weight_of_stored_pattern= weight_of_stored_pattern/5

#print(weight_variable)
print(weight_of_stored_pattern)

def feeding_patern(input):
  flag= False
  new_state= input.copy()
  for i in range(len(input)):
    new_state[i]=0
    for j in range (len(input)):
      new_state[i] = new_state[i]+ weight_of_stored_pattern[i][j]*new_state[j]
    if new_state[i] >=0:
      new_state[i] = 1
    elif new_state[i] < 0:
      new_state[i] = (-1)
    if np.array_equal(new_state,weight_variable.values()):
      flag= True
      print (flag)
      return (new_state, flag)
    input= new_state
  return new_state

'''

def feeding_pattern (input):

  while (new_state in weight_variable.values()):
  for t in range (0, 1000):
    new_state = np.sign(np.dot(weight_of_stored_pattern*input))
    if new_state  in weight_variable.values():
      return (new_state)
    input=new_state.copy()
  return(new_state)
'''

#enter the pattern here
input_pattern = [[1, -1, 1, 1, 1, -1, 1, 1, -1, -1], [1, -1, 1, 1, 1, -1, 1, 1, 1, -1], [1, -1, -1, -1, -1, 1, 1, 1, 1, -1], [1, -1, -1, -1, -1, 1, 1, 1, 1, -1], [1, -1, -1, -1, -1, 1, 1, 1, 1, -1], [1, -1, -1, -1, -1, 1, 1, 1, 1, -1], [1, -1, -1, -1, -1, 1, 1, 1, 1, -1], [1, -1, 1, 1, 1, -1, 1, 1, -1, -1], [1, -1, 1, 1, 1, -1, 1, 1, -1, -1], [1, -1, -1, -1, -1, 1, 1, 1, 1, -1], [1, -1, -1, -1, -1, 1, 1, 1, 1, -1], [1, -1, -1, -1, -1, 1, 1, 1, 1, -1], [1, -1, -1, -1, -1, 1, 1, 1, 1, -1], [1, -1, -1, -1, -1, 1, 1, 1, 1, -1], [1, -1, 1, 1, 1, -1, 1, 1, 1, -1], [1, -1, 1, 1, 1, -1, 1, 1, -1, -1]]

#input('enter the pattern')
input_pattern_column = matrix_to_column(input_pattern)
#print(input_pattern_column)
result = np.array(feeding_patern(input_pattern_column))

result1 = result.reshape(16,10)
result2 = formatted_matrix = "[" + ",\n".join(["[" + ",".join(map(str, row)) + "]" for row in result1]) + "]"

print(result2)


def inverse_prediction (m): #fun to find the inverse of letter
  a1, a2= 1,-1
  for i in range (len(result1)):
    for j in range (len(result1[0])):
      if m[i][j] == a1:
        m[i][j] == a2
      else:
        if m[i][j]==a2:
          m[i][j] == a1
  return m
print('Pattern coverges to:')
if np.array_equal(result1,x1):
  print ("x1")
elif np.array_equal(result1,x2):
  print ("x2")
elif np.array_equal(result1,x3):
  print ("x3")
elif np.array_equal(result1,x4):
  print ("x4")
elif np.array_equal(result1,x5):
  print ("x5")
else:
  inverse = inverse_prediction(result1)
  if np.array_equal(inverse,x1):
    print ("-x1")
  elif np.array_equal(inverse,x2):
    print ("-x2")
  elif np.array_equal(inverse,x3):
    print ("-x3")
  elif np.array_equal(inverse,x4):
    print ("-x4")
  elif np.array_equal(inverse,x5):
    print ("-x5")