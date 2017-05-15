[Gym]: https://gym.openai.com/ 'OpenAI Gym'
[PLE]: https://github.com/ntasfi/PyGame-Learning-Environment 'PLE'
[dqn_agent_play]: ./09242016_DQN_Snake.gif 'DQN agent playing snake game'

## Snake Game by played by Reinforcement Learning (DQN)

![alt text][dqn_agent_play]

### Why PLE
Branched off from [PLE], with local changes/'hacks' with Python3 and snake game. *Didn't try other included games at all*. 

This was created before OpenAI's [Gym] was created. The benefit of PLE is you can add your own PyGame based game onto the environment side (I didn't find Snake in OpenAI's zoo). However, with Gym simulator available, the PLE work (IMHO) might not worth further investigation thus I stopped. 

### Required packages
* Ubuntu 16.04 LTS
* Python 3
* Pygame (>=v1.9.2, which works with Python3). Can be installed with 
```sh
pip3 install pygame
```
* Tensorflow 1.0 

### How to play

* Set up environment variable $PYTHONPATH
```sh
export PYTHONPATH = <where your PyGame-Learning-Environments dir>
```

* Play by human player (Keys ASDW for directions)
```sh
cd PyGame-Learning-Environments/examples
python3 snake_human_play.py 
```

* Play with an agent issuing random commands
```sh
python3 snake_random_agent.py 
```

* Training with DQN agent
```sh
python3 snake_dqn_agent.py 
```

* Play with trained model 
```sh
python3 snake_dqn_agent_review.py 
```
