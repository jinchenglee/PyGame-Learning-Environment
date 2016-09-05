from ple.games import Snake
from ple import PLE
import numpy as np
import random
from collections import deque

#---------------------------------------------------------------
# DQN Parameters
#---------------------------------------------------------------
class DQN_Agent():
    """
    Deep-Q Network agent class.
    """
    def __init__(self, actions):
        self.actions = list(actions)

    def create_DQN(self):
        pass

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
EPISODE = 201 # Episode limit
STEP = 100      # Step limit within one episode 
TEST = 10       # Number of experiement test every 100 episode

def main():
    # Init PygameLearningEnv env and agent
    game = Snake(width=16, height=16)
    env = PLE(game, fps=60, display_screen=True, force_fps=True, add_noop_action=False)
    agent = DQN_Agent(actions=env.getActionSet())
    
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
