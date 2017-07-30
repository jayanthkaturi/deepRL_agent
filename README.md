## Deep Reinforcement Learning Agent from scratch

Reinforcement learning with the power of Deep learning is proving to be a very good in driving cars or playing atari games. In this post I'll try to explain how this works using a basic example of moving a robot in 2D matrix space. You'll see how a DeepRL agent learned, all by it self, to manoeuvre in zig-zag pattern so that it'll receive more rewards. I am using a neural network which I implemented from scratch for this. I'll use term “robot” for this DeepRL agent.

I won't get into much of neural networks, but I'll explain how I implemented them. There are several courses/tutorials online that explains how these work in very detail. The neural network is stored as a numpy array of layers. Each layer has a numpy array of nodes in that layer. Each node is a dictionary with weights from previous layer, bias, delta on this node (used while backpropagation), output of this node and delta in bias. The class NN takes an array with configuration of the neural network as input. [2,3,4,5] represents 2 input nodes, 2 hidden layers with 3 and 4 nodes each and 5 output nodes. The activation function used is sigmoid as we are trying to predict a single value (we'll get to this). The configuration which I used for this example is [3, 10, 10, 1]. I used root mean square error as loss funciton. The backward function returns this error and forward function returns the output for given input. PrintNN function just prints the network state for understanding. To run the code, do the following. Numpy is the only dependency needed.
```bash
pip install numpy
python moveRobot.py
```

For a Reinforcement problem, we need states, actions and rewards. The state here is the position of the robot in a 2D cor-ordinate space. Simultaneously I am maintaining another 2D matrix of environment with the position of robot as 1 and rest of the entries 0. (I am calling this as state and the former one as position of robot in the code).
```
                            [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
                             [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
                             [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
                             [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
                             [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
                             [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
                             [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
                             [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
                             [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
                             [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.]
                             [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
                            This state represents robot at [9,9] in the environment.
```
The number of possible actions are 4, i.e, the robot can go forward in any direction and backward only in down direction. Each of these actions are given numbers from 5 to 8. We are allowing the robot only to move forward because it gets stuck in local optima if it allowed to move backwards. This means the robot is going to move back and forth in a loop in the field, in which case it'll get more rewards. But this doesn't bring the robot to the end of the matrix. Hence this restriction. You can experiment with increasing states by including few backward moments and see what happens. The matrix below shows the action number corresponding to the movement of robot from cell 'p'.
```
                                              [[1 2 3]
                                               [4 p 5]
                                               [6 7 8]]
                                            Action numbers
```
The act(state, pos_robot, action, possibility=False) takes the 2D state matix like on the 2D marix above, the position of robot, the action number and an optional possibility variable. If possiblity is set to True, the output is just True or False if the action is possible (it is not possible if the action takes robot out of the 2D matrix). Else, it'll update the state matrix and return the new position of the robot along with the new state. The rewards are also a 2D matrix representing how much the robot will receive if it reaches a cell as shown below. The numbers are less than one because we are using sigmoid activation for output layers and as you'll see later, this reward is directly related to the expected output of the neural network.
```
                    [[ 0.    0.05  0.1   0.15  0.2   0.25  0.3   0.35  0.4   0.45  0.5 ]
                     [ 0.05  0.    0.15  0.2   0.25  0.3   0.35  0.4   0.45  0.5   0.55]
                     [ 0.1   0.15  0.    0.25  0.3   0.35  0.4   0.45  0.5   0.55  0.6 ]
                     [ 0.15  0.2   0.25  0.    0.35  0.4   0.45  0.5   0.55  0.6   0.65]
                     [ 0.2   0.25  0.3   0.35  0.    0.45  0.5   0.55  0.6   0.65  0.7 ]
                     [ 0.25  0.3   0.35  0.4   0.45  0.    0.55  0.6   0.65  0.7   0.75]
                     [ 0.3   0.35  0.4   0.45  0.5   0.55  0.    0.65  0.7   0.75  0.8 ]
                     [ 0.35  0.4   0.45  0.5   0.55  0.6   0.65  0.    0.75  0.8   0.85]
                     [ 0.4   0.45  0.5   0.55  0.6   0.65  0.7   0.75  0.    0.85  0.9 ]
                     [ 0.45  0.5   0.55  0.6   0.65  0.7   0.75  0.8   0.85  0.    0.95]
                     [ 0.5   0.55  0.6   0.65  0.7   0.75  0.8   0.85  0.9   0.95  0.  ]]
                      Each cell representing the reward it'll receive for reaching it.
```
The aim of the robot is to go from [0,0] to [10,10] collecting maximum rewards on the way. (This can be done with a DP but the point is to understand how deep RL works.)  I am also using Replay memory. This means that the previous actions are stored in the form (initial state, initial position , action, final state, final position) so that they can be processed it later. This is used in order to avoid the robot to get stuck in local optimum. The algorithm is as follows :
1. Choose a random action and execute it on the current state (act() function does this after action is choosen randomly).
2. Store this action in memory. This is the replay memory. The size is fixed to 10. If there it exceeds 10, the entry in the beginning will be removed.
3. Choose a random entry in the replay memory.
4. If that memory has reached the end, the input to neural network will be a vector of length 3 with position in the first two entries and action number in the third. The output will be the reward it'll receive.
5. Else, for the final position of this entry, find an entry that maximizes the final reward. This is done by iterating through all the actions with the first two entries of the input of neural network as the final position in this entry of replay memory.
6. After recieving the action that maximizes the reward, set the input of the neural network as initial position and action of this entry of replay memory and output as  (Reward + 0.5 * maximum reward we obtained from before step) 

This is repeated for every epoch and time step. What we tried to do here is get the optimal Q-function using neural network. The equaiton looks like this
```
                             Q(s,a) = r + (gamma * max(Q(s',a')))
```
This Q is a function of states and actions. It's output value is the optimal reward for given state s and action a taken. s' is the final state in a replay memory and max funtion is to get the maximum Q value for all possible actions from state s'. 'gamma' is similar to learning rate in neural networks. For our problem, we need a 100x8 matrix if we decided to do this using just reinforcement learning. What the neural nerwok did here is replace this giant function using a neural network. This problem might look small but to do this on an [Atari Game](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) we need about math.pow(256, 84 * 84 * 4) matrix!. Thus a neural network, which is efficient in learning functions does this without having to store the Q values of all the states and actions. 

By running moverobot.py, one can see how the robot is moved through optimal path.
```
(0,0) -> (0,1) -> (0,2) -> (0,3) ->...-> (0,10) -> (1,9) ->  (1, 10) ->
(2,9) -> (2,10) -> (3,9) -> (3,10) .... -> (9,9) -> (9,10) -> (10, 9) -> (10, 10)
```
Remember the only input to this neural network is state, action and output is reward it'll receive. Using just this much the neural network has learned how to move through the environment to receive maximum reward. It learned that the concentration of maximum rewards is towards the right hand corner and it also learned that given the restrictions to move only forward, going zigzag will give it more rewards than going in straight line! Amazing isn't it? The output of the code prints the reward matrix first, neural network architecture and each of the steps the learned neural network took in order to reach the output. Each step consists of the input ([position action]), output of neural network and state matrix after the action is performed.
