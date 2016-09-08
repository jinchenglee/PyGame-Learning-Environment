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
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())


    def create_DQN(self):
        # network weights
        W1 = self.weight_variable([self.state_dim,HIDDEN_LAYER_DEPTH])
        b1 = self.bias_variable([HIDDEN_LAYER_DEPTH])
        W2 = self.weight_variable([HIDDEN_LAYER_DEPTH,self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        # input layer
        self.state_input = tf.placeholder(tf.float32, [None, self.state_dim])
        print("reshaped dim: ", self.state_input)
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

    def create_training_method(self):
        pass

    def perceive(self, observation, action, reward, next_observation, done):
        pass

    def train_DQN(self):
        pass

    def pickAction(self, observation):
        pass

    def pickAction_egreedy(self, observation):
        pass



#---------------------------------------------------------------
# Hyper Parameters
#---------------------------------------------------------------
EPISODE = 1 # Episode limit
STEP = 10      # Step limit within one episode 
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
    
        # Training process
        for step_train in range(STEP):
            # Explore, use egreedy version
            action = agent.pickAction_egreedy(observation)
    
            reward = env.act(action)
            next_observation = env.getScreenRGB()
            done = env.game_over()
    
            agent.perceive(observation, action, reward, next_observation, done)

            observation = next_observation
    
            #print(observation)
            #env.savescreen("tmp/image"+str(step_train).zfill(5)+".png")

            if done:
                break

        # Test every 100 espisodes
        if episode % 100 == 0:
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
