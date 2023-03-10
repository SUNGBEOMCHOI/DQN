import copy
from collections import deque

import numpy as np
import gym
import imageio
from PIL import Image, ImageDraw, ImageFont

from torchvision.transforms import transforms as T
from torchvision.transforms.functional import crop

class Env:
    def __init__(self):
        pass

    @staticmethod
    def make(env_name, env_cfg=None):
        if env_name == 'CartPole':
            return CartPole(env_cfg)
        elif env_name == 'Breakout':
            return Breakout(env_cfg)

class CartPole(gym.Env):
    metadata = {'render.modes': ['rgb_array', 'human']}

    def __init__(self, env_cfg=None):
        self.env = gym.make('CartPole-v1')
        self.model = None
        self.total_reward = 0.0
        self.timestep = 0
        self.is_save_video = False
        self.video_path = ''
        self.frames = []

    def step(self, action):
        if self.is_save_video:
            image = self.render(mode='rgb_array')
            image = self.edit_image(image, action)
            self.frames.append(image)

        self.timestep += 1
        state, reward, done, info = self.env.step(action)
        self.total_reward += reward
        if done:
            state = state = np.zeros((4), dtype=np.float32)
            if self.timestep!=500:
                reward = -1.0
            else:
                reward = 1.0
        else:
            reward = 0.1

        if self.is_save_video and done:
            with imageio.get_writer(self.video_path, fps=30, macro_block_size = None) as video:
                for frame in self.frames:
                    video.append_data(frame)

        return state, reward, done, info
    
    def reset(self):
        self.model = None
        self.total_reward = 0.0
        self.timestep = 0
        self.is_save_video = False
        self.video_path = ''
        self.frames = []
        return self.env.reset()

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self.env.render(mode='rgb_array')
        elif mode == 'human':
            return self.env.render(mode='human')
        else:
            raise NotImplementedError

    def save_video(self, model=None, video_path='./video/test.mp4', ):
        self.is_save_video = True
        self.video_path = video_path
        self.model = model

    def edit_image(self, image, action):
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.truetype("arial.ttf", 36)
        text = f'Q value : {self.model.q_value[action].item():.2f}'
        draw.text((10, 10), text, font=font, fill=(255, 0, 0)) # red color
        return np.array(pil_image)



# breakout ??????
# ????????? ???????????? ?????? ?????? ??????
class Breakout(gym.Env):
    metadata = {'render.modes': ['rgb_array', 'human']}

    def __init__(self, env_cfg=None):
        self.env = gym.make('ALE/Breakout-v5')
        self.model = None
        self.total_reward = 0.0
        self.timestep = 0
        self.process = None
        self.is_save_video = False
        self.video_path = ''
        self.frames = []
        self.four_frames = deque(maxlen=4)

    def step(self, action):
        self.timestep += 1   
        
        state, reward, done, info = self.env.step(action)
        self.four_frames.append(state)
        self.frames.append(self.render())
        self.total_reward += reward

        if done:
            state = np.zeros((4, 84, 84), dtype=np.float32)
        else:
            state = self.preprocessing(self.four_frames)

        if self.is_save_video and done:
            with imageio.get_writer(self.video_path, fps=30, macro_block_size = None) as video:
                for frame in self.frames:
                    video.append_data(frame)

        return state, reward, done, info
    
    def reset(self):
        self.total_reward = 0.0
        self.timestep = 0
        self.is_save_video = False
        self.video_path = ''
        self.frames = []

        state = self.env.reset()
        self.frames.append(self.render())
        self.four_frames.append(state)
        for _ in range(3):
            state, reward, _, _ = self.env.step(0)
            self.four_frames.append(state)
            self.total_reward += reward
            self.frames.append(self.render())

        state = self.preprocessing(self.four_frames)
        return state

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self.env.render(mode='rgb_array')
        elif mode == 'human':
            return self.env.render(mode='human')
        else:
            raise NotImplementedError

    def save_video(self, model=None, video_path='./video/test.mp4', ):
        self.is_save_video = True
        self.video_path = video_path
        self.model = model

    def edit_image(self, image, action):
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.truetype("arial.ttf", 36)
        text = f'Q value : {self.model.q_value[action].item():.2f}'
        draw.text((10, 10), text, font=font, fill=(255, 0, 0)) # red color
        return np.array(pil_image)

    def preprocessing(self, state):
        def bottom_crop(image):
            return crop(image, 110-84, 0, 84, 84)

        if self.process == None:
            self.process = T.Compose([T.ToPILImage(),
                        T.Grayscale(), # RGB -> gray scale
                        T.Resize((110,84)),
                        T.Lambda(bottom_crop), # image bottom crop
                        T.ToTensor()]) # pixel value 0~255 -> 0~1

        state = np.array([self.process(single_state).squeeze(0).cpu().detach().numpy() for single_state in state])
        return state

if __name__ == "__main__":
    env = Env.make('Breakout')
    state = env.reset()
    done = False
    timestep = 0
    while not done:
        timestep += 1
        state, _, done, _ = env.step(0)
    print(timestep)
