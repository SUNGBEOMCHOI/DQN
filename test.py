import os
import argparse

import yaml
import torch

from models import Model
from env import Env

def test(args, cfg):
    """
    Test trained model
    """
    ########################
    #   Get configuration  #
    ########################
    device = torch.device('cuda' if cfg['device']=='cuda' and torch.cuda.is_available() else 'cpu')
    test_cfg = cfg['test']
    video_path = cfg['test']['video_path']
    model_cfg = cfg['model']
    env_name = cfg['env']['name']

    os.makedirs(video_path, exist_ok=True)

    ########################
    # Get pretrained model #
    ########################
    model = Model(model_cfg, device).to(device)
    if args.pretrained:
        checkpoint = torch.load(args.pretrained)
    else:
        checkpoint = torch.load(test_cfg['model_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    ########################
    #   Make Environment   #
    ########################
    env = Env.make(env_name)
    
    ########################
    #      Test model     #
    ########################
    done = False
    state = env.reset()
    env.save_video(model=model)
    while not done:
        with torch.no_grad():         
            action = model.get_action(state)
            q_value = model.q_value[action].item()
            print(q_value)
            # total_q_value += q_value
        state, _, done, _ = env.step(action)
    total_reward = env.total_reward
    
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/cartpole_config.yaml', help='Path to config file')
    parser.add_argument('--pretrained', type=str, default='', help='Path to pretrained model file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    test(args, cfg)