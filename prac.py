import gym
import yaml

# env = gym.make('CartPole-v1')

# env.reset()
# done = False
# i = 0
# while not done:
#     i = 0 if i == 1 else 1
#     env.render('human')
#     _, _, done, _ = env.step(i)

with open('./config/test.yaml') as f:
    cfg = yaml.safe_load(f)
    print(cfg)