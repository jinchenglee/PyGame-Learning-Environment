from ple.games import Snake 
from ple import PLE
import numpy as np

class NaiveAgent():
    """
            This is our naive agent. It picks actions at random!
    """
    def __init__(self, actions):
        self.actions = list(actions)
        #print(actions)
        self.index=0

    def pickAction(self, reward, obs):
        self.index = np.random.randint(0, len(self.actions))
        #self.index = (self.index+1)%4
        #print(self.index)
        return self.actions[self.index]



game = Snake(width=128, height=128)
p = PLE(game, fps=30, display_screen=True, force_fps=False, add_noop_action=False)
agent = NaiveAgent(actions=p.getActionSet())

p.init()
reward = 0.0
total_reward = 0.0
nb_frames = 200
observation = p.getScreenRGB()
done = False

for i in range(nb_frames):
    if done:
        p.reset_game()
        reward = 0.0
        total_reward = 0.0

    action = agent.pickAction(reward, observation)

    reward = p.act(action)
    observation = p.getScreenRGB()
    done = p.game_over()

    total_reward += reward

    #print(observation)
    #p.savescreen("tmp/image"+str(i).zfill(5)+".png")
    print(total_reward, reward)





