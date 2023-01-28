import os
import argparse
import copy
from collections import deque

import yaml
import numpy as np
import torch
import gym

from models import Model
from env import Env
from utils import loss_func, optim_func, plot_progress, save_model

def train(args, cfg):
    '''
    Train model
    '''
    ########################
    #   Get configuration  #
    ########################
    device = torch.device('cuda' if cfg['device']=='cuda' and torch.cuda.is_available() else 'cpu')
    train_cfg = cfg['train']
    batch_size = train_cfg['batch_size']
    train_epochs = train_cfg['train_epochs']
    loss_name_list = train_cfg['loss']
    optim_cfg = train_cfg['optim']
    max_eps = train_cfg['max_eps']
    min_eps = train_cfg['min_eps']
    eps_decay = train_cfg['eps_decay']
    discount_factor = train_cfg['discount_factor']
    target_update_period = train_cfg['target_update_period']
    target_update_ratio = train_cfg['target_update_ratio']
    memory_length = train_cfg['memory_length']
    model_path = train_cfg['model_path']
    progress_path = train_cfg['progress_path']
    plot_epochs = train_cfg['plot_epochs']
    model_cfg = cfg['model']
    env_name = cfg['env']['name']

    ########################
    #      Make model      #
    ########################
    policy_net = Model(model_cfg, device).to(device)
    target_net = Model(model_cfg, device).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    ########################
    #   Make Environment   #
    ########################
    env = Env.make(env_name)

    ########################
    #    train settings    #
    ########################
    replay = {'state':deque(maxlen=memory_length), 
                'action':deque(maxlen=memory_length),
                'reward':deque(maxlen=memory_length),
                'next_state':deque(maxlen=memory_length),
                'done':deque(maxlen=memory_length)}
    criterion_list = loss_func(loss_name_list)
    optimizer = optim_func(policy_net, optim_cfg)
    history = {'loss':[], 'score':[], 'q_value':[]} # for saving loss
    start_epoch = 1

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(progress_path, exist_ok=True)

    if args.resume: # if pretrained model exists
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']+1
        policy_net.load_state_dict(checkpoint['model_state_dict'])
        history = checkpoint['history']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        target_net.load_state_dict(policy_net.state_dict())


    ########################
    #      Train model     #
    ########################
    for epoch in range(start_epoch, train_epochs+1):
        total_loss = 0.0
        state = env.reset()
        if epoch % 10 == 1:
            env.save_video(model=policy_net, video_path='./video/breakout.mp4')
        done, time_step = False, 0
        eps = max(max_eps-epoch*((max_eps-min_eps)/eps_decay), min_eps)
        while not done:
            time_step += 1
            if eps < np.random.rand():
                action = policy_net.get_action(state)
            else:
                action = np.random.choice(2)
            next_state, reward, done, _ = env.step(action)

            replay['state'].append(copy.deepcopy(state))
            replay['action'].append(action)
            replay['reward'].append(reward)
            replay['next_state'].append(copy.deepcopy(next_state))
            replay['done'].append(done)            
            
            state = next_state
            if len(replay['state']) > batch_size:
                idxs = np.random.choice(len(replay['state']), batch_size)
                states = torch.tensor(np.array(replay['state']))[idxs]
                actions = torch.tensor(replay['action'])[idxs]
                rewards = torch.tensor(replay['reward'])[idxs]
                next_states = torch.tensor(np.array(replay['next_state']))[idxs]
                dones = torch.tensor(replay['done'])[idxs]                

                target_q = torch.zeros(batch_size)
                target_q[dones] = rewards[dones]
                not_done_states = next_states[~dones]
                target_q[~dones] = discount_factor*torch.amax(target_net(not_done_states), dim=-1) + rewards[~dones]

                pred_q = torch.gather(policy_net(states.detach()), dim=1, index=actions.unsqueeze(-1).detach()).squeeze(-1)
                loss = criterion_list[0](pred_q, target_q.detach())
                optimizer.zero_grad()                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                
        # if epoch % target_update_period == 0:
        #     target_net.load_state_dict(policy_net.state_dict())
                for params in zip(policy_net.state_dict().values(), target_net.state_dict().values()):
                    value_param, target_value_param = params
                    new_target_value_param = (1-target_update_ratio)*target_value_param + target_update_ratio*value_param
                    target_value_param.copy_(new_target_value_param)

        history['loss'].append(total_loss/time_step)
        if epoch > 50:
            total_reward, q_value = validation(env, policy_net)
        else:
            total_reward, q_value = env.total_reward, 0.0
        history['score'].append(total_reward)
        history['q_value'].append(q_value)
        print(f'------ {epoch:03d} training ------- train loss : {total_loss/time_step:2.6f} -------- total reward : {int(total_reward):3d} ------- q value : {q_value:3.3f} -------')
        if epoch % plot_epochs == 0:
            plot_progress(history, epoch, progress_path)
            save_model(epoch, policy_net, optimizer, history, model_path)
            pass
        

def validation(env, model):
    total_q_value = 0.0
    done = False
    state = env.reset()
    # env.save_video(model=model, video_path='./video/breakout.mp4')
    while not done:
        with torch.no_grad():
            action = model.get_action(state)
            total_q_value += model.q_value[action].item()
        state, _, done, _ = env.step(action)
    total_reward = env.total_reward
    return total_reward, total_q_value/(env.timestep)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/breakout_config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default='', help='Path to pretrained model file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(args, cfg)