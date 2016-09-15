from ple.games import Snake
from ple import PLE
import numpy as np
import random
from collections import deque
import tensorflow as tf

#---------------------------------------------------------------
# DQN Parameters
#---------------------------------------------------------------
# Hyper Parameters for DQN
NUM_CHANNELS = 3
HIDDEN_LAYER_DEPTH = 512

GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 64 # size of minibatch
FRAME_BUF_DEPTH = 4 # number of frames to keep in history

class DQN_Agent():
    """
    Deep-Q Network agent class.
    """
    def __init__(self, env):
        # init experience buffer
        self.replay_buffer = deque()
        
        # All valid actions for the game
        self.actions = list(env.getActionSet())
        self.action_dim = len(self.actions)
        print("valid actions list: ", self.actions, "action_dim: ", self.action_dim)
        
        # Image screen resolution 
        self.state_dim = list(env.getScreenDims())
        # History of frames to keep in training data
        self.state_dim.extend([FRAME_BUF_DEPTH])
        print("game screen dim: ", self.state_dim)

        # Some more parameters init
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON

        # Create network
        self.create_DQN_conv()
        self.create_tensorflow()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

        # Add ops to save and restore all the variables.
        #self.saver = tf.train.Saver()


    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, writh bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    # Convolutional NN impl.
    def create_DQN_conv(self):
        # network weights
        Wc1 = self.weight_variable([3,3,4,16])
        bc1 = self.bias_variable([16])
        Wc2 = self.weight_variable([3,3,16,32])
        bc2 = self.bias_variable([32])
        Wfc = self.weight_variable([self.state_dim[0]*self.state_dim[1]*32,HIDDEN_LAYER_DEPTH])
        bfc = self.bias_variable([HIDDEN_LAYER_DEPTH])
        Wout = self.weight_variable([HIDDEN_LAYER_DEPTH,self.action_dim])
        bout = self.bias_variable([self.action_dim])
        # input layer - [batch size, screen_X, screen_Y, num_frame]
        self.state_input = tf.placeholder(tf.float32, [None, self.state_dim[0],self.state_dim[1],self.state_dim[2]])
        #print("state_input shape: ", self.state_input)
        # conv layers
        conv_layer1 = self.conv2d(self.state_input,Wc1,bc1)
        conv_layer2 = self.conv2d(conv_layer1,Wc2,bc2)
        # Fully connected layer
        conv_layer2_flat = tf.reshape(conv_layer2,[-1,self.state_dim[0]*self.state_dim[1]*32])
        fc_layer = tf.nn.relu(tf.matmul(conv_layer2_flat,Wfc) + bfc)
        # Q Value layer
        self.Q_value = tf.matmul(fc_layer,Wout) + bout

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)


    def create_tensorflow(self):
        self.action_input = tf.placeholder(tf.float32, [None, self.action_dim]) # one-hot representation
        self.y_input = tf.placeholder(tf.float32, [None])
        Q_value_of_action = tf.reduce_sum(tf.mul(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_value_of_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self, observation, action, reward, next_observation, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[self.actions.index(action)] = 1
        #print(action, one_hot_action)
        self.replay_buffer.append((observation, one_hot_action, reward, next_observation, done))

        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_DQN()

    def train_DQN(self):
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
        # Create *observation_batch numpy array with right size
        observation_batch = np.empty([1,self.state_dim[0], self.state_dim[1], FRAME_BUF_DEPTH])
        next_observation_batch = np.empty([1,self.state_dim[0], self.state_dim[1], FRAME_BUF_DEPTH])
        action_batch = []
        reward_batch = []
        for data in minibatch:
            data_0 = data[0].reshape([1,self.state_dim[0], self.state_dim[1], FRAME_BUF_DEPTH])
            data_3 = data[3].reshape([1,self.state_dim[0], self.state_dim[1], FRAME_BUF_DEPTH])
            #print("observation_batch.shape = ", observation_batch.shape)
            #print("data[0].shape = ", data[0].shape)
            #print("data_0.shape = ", data_0.shape)
            observation_batch = np.concatenate((observation_batch, data_0), axis=0)
            action_batch.append(data[1])
            reward_batch.append(data[2]) 
            next_observation_batch = np.concatenate((next_observation_batch, data_3), axis=0)
        # Remove the randomly created top line of *observation_batch array
        observation_batch = np.delete(observation_batch,0,axis=0)
        next_observation_batch = np.delete(next_observation_batch,0,axis=0)

        # Step 2: calculate y (Q-value of action in real play)
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_observation_batch})
        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
                self.y_input:y_batch,
                self.action_input:action_batch,
                self.state_input:observation_batch
            })

    def saveParam(self):
        # Save the scene
        #save_path = self.saver.save(self.session, "./tmp/model_tr_"+str(self.time_step)+".ckpt")
        pass

    def restoreParam(self):
        # Restore the scene
        #self.saver.restore(self.session, "./tmp_e/model_tr_e.ckpt")
        pass

    def pickAction(self, observation):
        Q_value = self.Q_value.eval(feed_dict = { 
            self.state_input:[observation] 
            })[0]
        #print("pickAction: Estimated Q-Values = ", Q_value)
        action_index = np.argmax(Q_value)
        #print("pickAction: picked action = ", action_index)
        return self.actions[action_index]

    def pickAction_egreedy(self, observation):
        Q_value = self.Q_value.eval(feed_dict = { 
            self.state_input:[observation] 
            })[0] # Evaluation take in [N, self.action_dim], thus output N of Q-values. Here N=1.
        #print("pickAction_egreedy: Estimated Q-Values = ", Q_value)
        if random.random() <= self.epsilon:
            action_index = random.randint(0,self.action_dim - 1)
        else:
            action_index = np.argmax(Q_value)
        #print("pickAction_egreedy: picked action = ", action_index)
        return self.actions[action_index]



#---------------------------------------------------------------
# Hyper Parameters
#---------------------------------------------------------------
EPISODE = 50000 # Episode limit
STEP = 100      # Step limit within one episode 
TEST = 10       # Number of experiement test every 100 episode

def luminance(RGB):
    """
    Convert screen RGB into luminance. [screen_X, screen_Y, channel] -> [screen_X, screen_Y]
    """
    #print("RGB shape:", RGB.shape)
    lum = np.dot(RGB,[0.299,0.587,0.114])
    #print("lum = ", lum)
    return lum

def main():
    # Init PygameLearningEnv env and agent
    game = Snake(width=16, height=16)
    env = PLE(game, fps=60, display_screen=True, force_fps=True, add_noop_action=False)
    agent = DQN_Agent(env)
    
    for episode in range(EPISODE):
        # Initialize task
        env.init()
        env.reset_game()
        env.display_screen=False
        env.force_fps = True
        # TODO: FIXME
        observation = np.asarray([luminance(env.getScreenRGB())]*FRAME_BUF_DEPTH).reshape(16,16,FRAME_BUF_DEPTH)
        #print("observation shape = ", observation.shape)
        reward = 0.0
        done = False
    
        # Restore the trained parameters
        agent.restoreParam()

        # Training process
        for step_train in range(STEP):
            # Explore, use egreedy version
            action = agent.pickAction_egreedy(observation)
    
            reward = env.act(action)
            #print("step reward = ", reward)

            next_observation = np.zeros_like(observation)
            #print("next_observation shape = ", next_observation.shape)
            next_observation[:,:,1:] = observation[:,:,:-1]
            next_observation[:,:,0] = luminance(env.getScreenRGB())
            done = env.game_over()
            #dist = game.getSnakeheadFoodDistance()
    
            agent.perceive(observation, action, reward, next_observation, done)

            observation = next_observation
    
            #for i in range(FRAME_BUF_DEPTH):
            #    print("next_observation", i, " = ", next_observation[:,:,i])
            #print("next_observation", 0, " = ", next_observation[:,:,0])
            #env.savescreen("tmp/image"+str(step_train).zfill(5)+".png")

            if done:
                break

        # Test every 1000 espisodes
        if episode % 1000 == 0:
            total_reward = 0
            env.display_screen=True # <<JC>> View the test result?
            env.force_fps = False

            # Save the trained parameters
            agent.saveParam()

            for test in range(TEST):
                # Initialize task
                env.init()
                env.reset_game()
                # TODO: FIXME
                observation = np.asarray([luminance(env.getScreenRGB())]*FRAME_BUF_DEPTH).reshape(16,16,FRAME_BUF_DEPTH)
                reward = 0.0
                done = False
            
                # Training process
                for step_test in range(STEP):
                    # Use the non-greedy selection
                    action = agent.pickAction(observation)
            
                    reward = env.act(action)
                    observation[:,:,1:] = observation[:,:,:-1]
                    observation[:,:,0] = luminance(env.getScreenRGB())
                    done = env.game_over()
            
                    total_reward += reward
            
                    #print(observation)
                    #env.savescreen("tmp/image"+str(step_test).zfill(5)+".png")

                    if done:
                        break

            average_reward = total_reward/TEST
            print("episode: ", episode, "Evaluation Average Reward: ", average_reward)
            if average_reward >= 10:
                break


if __name__ == '__main__':
    main()
