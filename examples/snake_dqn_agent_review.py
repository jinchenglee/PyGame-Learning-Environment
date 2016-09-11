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
HIDDEN_LAYER_DEPTH = 1000

GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch

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
        self.state_dim_list = list(env.getScreenDims())
        self.state_dim_list.extend([NUM_CHANNELS])
        print("game screen dim: ", self.state_dim_list)
        self.state_dim = self.state_dim_list[0]*self.state_dim_list[1]*self.state_dim_list[2]

        # Some more parameters init
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON

        # Create network
        self.create_DQN()
        self.create_tensorflow()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()


    def create_DQN(self):
        # network weights
        W1 = self.weight_variable([self.state_dim,HIDDEN_LAYER_DEPTH])
        b1 = self.bias_variable([HIDDEN_LAYER_DEPTH])
        W2 = self.weight_variable([HIDDEN_LAYER_DEPTH,self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        # input layer
        self.state_input = tf.placeholder(tf.float32, [None, self.state_dim])
        #print("reshaped dim: ", self.state_input)
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
        # Q Value layer
        self.Q_value = tf.matmul(h_layer,W2) + b2

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
        observation_batch = np.empty([1,self.state_dim])
        next_observation_batch = np.empty([1,self.state_dim])
        action_batch = []
        reward_batch = []
        for data in minibatch:
            observation_batch = np.concatenate((observation_batch, data[0].reshape([1,self.state_dim])), axis=0)
            action_batch.append(data[1])
            reward_batch.append(data[2]) 
            next_observation_batch = np.concatenate((next_observation_batch, data[3].reshape([1,self.state_dim])), axis=0)
        # Remove the randomly created top line of *observation_batch array
        observation_batch = np.delete(observation_batch,0,axis=0)
        next_observation_batch = np.delete(next_observation_batch,0,axis=0)
        #print("next_observation_batch.shape = ", next_observation_batch.shape)

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
        self.saver.save(self.session, "./tmp/model_tr_"+str(self.time_step)+".ckpt")

    def restoreParam(self):
        # Restore the scene
        self.saver.restore(self.session, "./tmp_e/model_tr_e.ckpt")

    def pickAction(self, observation):
        Q_value = self.Q_value.eval(feed_dict = { 
            self.state_input:[observation.reshape(self.state_dim)] 
            })[0]
        #print("pickAction: Estimated Q-Values = ", Q_value)
        action_index = np.argmax(Q_value)
        #print("pickAction: picked action = ", action_index)
        return self.actions[action_index]

    def pickAction_egreedy(self, observation):
        Q_value = self.Q_value.eval(feed_dict = { 
            self.state_input:[observation.reshape(self.state_dim)] 
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
EPISODE = 1     # Episode limit
STEP = 300      # Step limit within one episode 
TEST = 10       # Number of experiement test every 100 episode

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
        observation = env.getScreenRGB()
        reward = 0.0
        done = False
    
        # Restore the trained parameters
        agent.restoreParam()

        #print(observation)
        #env.savescreen("tmp/image"+str(step_train).zfill(5)+".png")

        # Test every 100 espisodes
        total_reward = 0
        env.display_screen=True
        env.force_fps = False

        for test in range(TEST):
            # Initialize task
            env.init()
            env.reset_game()
            observation = env.getScreenRGB()
            reward = 0.0
            done = False
        
            # Training process
            for step_test in range(STEP):
                # Use the non-greedy selection
                action = agent.pickAction(observation)
        
                reward = env.act(action)
                observation = env.getScreenRGB()
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
