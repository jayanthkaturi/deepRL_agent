import numpy as np
import sys
from math import exp,sqrt
from random import randint
import copy
def activate(x):
	return 1.0/(1.0+exp(x))
	if x > 0:
		return x
	return 0
def dactivate(x):
	return x * (1.0 - x)
	if x > 0:
		return 1
	return 0
class NN:
	nn = []
	learning_rate = 0.1
	def __init__(self, configuration, learning_rate=0.1):
		for i in range(1, len(configuration)):
			self.nn.append(
				np.array([
				{
					'weights':np.array([(np.random.random() * sqrt(2.0/configuration[i])) for k in range(configuration[i-1])]),
					'bias':0.01,
					'dbias':0.0,
					'delta':0.0,
					'output':0.0
				}
				for j in range(configuration[i])
			]))
		self.nn = np.array(self.nn)
		self.learning_rate = learning_rate
	def forward(self, inp):
		nn = self.nn
		curr_inp = inp
		for layer in nn:
			temp_inp = []
			for node in layer:
				node['output'] = activate(np.sum(node['weights'] * curr_inp) + node['bias'])
				temp_inp.append(node['output'])
			curr_inp = np.array(temp_inp)
		self.nn = nn
		return curr_inp
	def backward(self, inp, out):
		nn = self.nn
		for idn,node in enumerate(nn[len(nn)-1]):
			node['delta'] = (node['output'] - out[idn]) * dactivate(node['output'])
			node['dbias'] = (node['output'] - out[idn])
		for idl in reversed(range(len(nn) - 1)):
			for idn,node in enumerate(nn[idl]):
				node['delta'] = dactivate(node['output']) * np.sum([(nextLayerNode['weights'][idn] * nextLayerNode['delta']) for nextLayerNode in nn[idl+1]])
				node['dbias'] = np.sum([nextLayerNode['dbias'] for nextLayerNode in nn[idl+1]])
		curr_inp = inp
		for layer in nn:
			temp_inp = []
			for node in layer:
				node['weights'] += (-1.0 * self.learning_rate * node['delta'] * curr_inp)
				node['bias'] += (self.learning_rate * node['dbias'])
				temp_inp.append(node['output'])
			curr_inp = np.array(temp_inp)
		self.nn = nn
		error = np.sum([(out[idn] - node['output'])**2 for idn, node in enumerate(nn[len(nn) - 1])])
		return error
	def printNN(self):
		for idl,layer in enumerate(self.nn):
			print("Layer %s"%str(idl+1))
			for node in layer:
				print(node)
"""
## NN test, try to over fit one example
nn = NN([2,4,2])
nn.printNN()
for i in range(100):
	if i%2 == 0:
		x = np.array([1,0])
		y = np.array([1,0])
	else:
		x = np.array([0,1])
		y = np.array([0,1])
	if i%3 == 0:
		print("epoch %s"%str(i))
		print(nn.forward(x))
		print(nn.backward(x,y))
		print(x,y)
	else:
		nn.forward(x)
		nn.backward(x,y)
print(nn.forward(np.array([1,0])))
print(nn.forward(np.array([0,1])))
"""
def act(state, pos_robot, action, possibility=False):
	if action == 0:
		return state, pos_robot, False
	temp_pos = [pos_robot[0],pos_robot[1]]
	if action == 1:
		temp_pos = [pos_robot[0]-1, pos_robot[1]-1]
	elif action == 2:
		temp_pos = [pos_robot[0]-1, pos_robot[1]]
	elif action == 3:
		temp_pos = [pos_robot[0]-1, pos_robot[1]+1]
	elif action == 4:
		temp_pos = [pos_robot[0], pos_robot[1]-1]
	elif action == 5:
		temp_pos = [pos_robot[0], pos_robot[1]+1]
	elif action == 6:
		temp_pos = [pos_robot[0]+1, pos_robot[1]-1]
	elif action == 7:
		temp_pos = [pos_robot[0]+1, pos_robot[1]]
	else:
		temp_pos = [pos_robot[0]+1, pos_robot[1]+1]
	if temp_pos[0] < 0 or temp_pos[0] >= len(state) or temp_pos[1] < 0 or temp_pos[1] >= len(state):
		if possibility:
			return False
		return state, pos_robot, False
	if possibility:
		return True
	state[pos_robot[0],pos_robot[1]] = 0
	state[temp_pos[0],temp_pos[1]] = 1
	return state, temp_pos, True
	
## set robot reward field
reward_field = np.zeros([11,11])
reward_field = np.array([np.array([((idx+idy)/20.0) for idy in range(len(reward_field[idx]))]) for idx in range(len(reward_field))])
for i in range(11):
	reward_field[i][i] = 0.0
print(reward_field)
num_actions = 8 ## move one step in any direction
curr_state = np.zeros([11,11]) ## state is a 2d array with position of robot as 1
curr_state[0][0] = 1 ## start from top left, destination bottom right
pos_robot = [0,0] ## keep track of robot
replay_mem = [] ## replay memory
nn = NN([3, 10, 10, 1])
for epoch in range(500):
	for time in range(10):
		## perform action and store in replay memory 
		possible = False
		prev_state = copy.deepcopy(curr_state)
		prev_pos_robot = copy.deepcopy(pos_robot)
		breaker = 0
		while not possible:
			action = randint(5,num_actions)
			curr_state, pos_robot, possible = act(curr_state, pos_robot, action)
			breaker += 1
			if breaker == 6:
				break
		if breaker == 6:
			continue
		if len(replay_mem) > 10:
			replay_mem = replay_mem[1:]
		replay_mem.append([
			copy.deepcopy(prev_state), 
			copy.deepcopy(prev_pos_robot), 
			action, 
			copy.deepcopy(curr_state), 
			copy.deepcopy(pos_robot), 
			reward_field[prev_pos_robot[0],prev_pos_robot[1]]
		])
		del prev_state
		del prev_pos_robot

		## choose random memory and train neural network
		mem = randint(0,len(replay_mem)-1)
		curr_mem = copy.deepcopy(replay_mem[mem])
		if curr_mem[4][0] == 10 and curr_mem[4][1] == 10:
			inp_nn = copy.deepcopy(curr_mem[1])
			inp_nn.append(curr_mem[2])
			inp_nn = np.array(inp_nn)
			nn.forward(inp_nn)
			nn.backward(inp_nn,np.array([curr_mem[5]]))
		else:
			max_reward = -1
			inp_nn = copy.deepcopy(curr_mem[4])
			inp_nn.append(0)
			inp_nn = np.array(inp_nn)
			action_taken = 0
			for i in range(num_actions):
				inp_nn[-1] = i+1
				curr_q_value = nn.forward(inp_nn)
				if curr_q_value[0] > max_reward:
					max_reward = curr_q_value[0]
					action_taken = i+1
			exp_output = curr_mem[5] + (0.5 * max_reward)
			if exp_output > 1.0:
				exp_output = 1.0
			elif exp_output < -1.0:
				exp_output = -1.0
			inp_nn = copy.deepcopy(curr_mem[1])
			inp_nn.append(curr_mem[2])
			inp_nn = np.array(inp_nn)
			nn.backward(inp_nn,np.array([exp_output]))

nn.printNN()
## Test robot
state = np.zeros([11,11])
state[0][0] = 1
curr_pos = [0,0]
while True:
	## For given state find the maximum Q valued action and perform it.
	## If it results in desired target state, it's success else failed.
	max_reward = -1
	action_to_take = 0
	inp = copy.deepcopy(curr_pos)
	inp.append(0)
	inp = np.array(inp)
	for i in range(4,num_actions):
		possible = act(state, curr_pos, (i+1), possibility=True)
		if not possible:
			continue
		inp[-1] = i+1
		curr_reward = nn.forward(inp)
		print(inp, curr_reward)
		if curr_reward[0] > max_reward:
			action_to_take = i+1
			max_reward = curr_reward[0]
	state, curr_pos, possible = act(state, curr_pos, action_to_take)
	print(state)
	if possible == False:
		print("Fail")
		sys.exit(0)
	if curr_pos[0] == 10 and curr_pos[1] == 10:
		print("Success!!")
		sys.exit(0)
