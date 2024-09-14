import sys 
sys.path.append("..") 
import json
import pygame
import numpy as np
from rl_agents import ppo
import random
import torch
import env.env as env
import params
import rl_agents.replay_buffer as replay_buffer
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from typing import Type
import logging

logger = logging.getLogger(__name__)

class BasePipe():
    def send(self, data):
        ...
    def recv(self):
        ...


class SubRenderWindow():
    """一个用于子进程中的渲染类"""
    def __init__(self, param:params.EnvParams=None) -> None:
        self.display:pygame.Surface = None
        if param is None:
            self.image_size = (700, 600)
        else:
            self.image_size = param.image_size
            

    def show(self, surf):
        """唤起窗口渲染画面, 返回窗口是否被用户关闭的句柄"""
        if self.display is None:
            self.display = pygame.display.set_mode(self.image_size)
        surf = surf if isinstance(surf, pygame.Surface) else pygame.surfarray.make_surface(surf)
        
        self.display.blit(surf, (0,0))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return True
        return False
    
    def close(self):
        if self.display is not None:
            pygame.display.quit()
            self.display = None


class SubEnv:
    """管理每个独立进程中的游戏环境"""
    def __init__(self, _env:env.BaseEnv, pram:params.EnvParams, id=None):
        self.id = id
        self.env = _env
        self.env.set_params(pram)
        self.window = SubRenderWindow()
        self.total_reward = 0
        self.total_step = 0

    # 重启环境
    def reset(self):
        state = self.env.reset()
        return state, None, None, None

    # 环境执行一步
    def step(self, a, render=False):
        # 环境任务如果已经完成则重置
        s_next, r, d, debug = self.env.step(a)
        self.total_reward += 1
        self.total_step += 1
        if self.total_step >= self.env.max_steps:
            d = True
        if render:
            image = self.env.render()
            self.window.show(image)

        if d:
            # logging.info(f"Env:{self.id}: Done with {self.total_reward} steps.")
            self.total_reward = 0
            self.total_step = 0
            self.env.close()
            self.window.close()
            self.env.reset()
            s_next = self.env.reset()
        return s_next, r, d, debug



class MultEnvStepper:
    def __init__(self, Env:Type[env.BaseEnv], 
                 env_prams:params.EnvParams, runtime_params:params.RuntimeParams):
        # 创建一堆通讯管道,每个子环境一个
        self.main2sub:list [BasePipe] = None 
        self.sub2main:list [BasePipe] = None
        self.replay_buffer = replay_buffer.ExperimentReplayBuffer()
        num_envs = runtime_params.num_processes
        self.Env = Env  # 类而非实例
        self.main2sub, self.sub2main = \
            zip(*[mp.Pipe() for _ in range(num_envs)])
        
        self.states = [None for _ in range(num_envs)]
        # 创建子环境的进程 
        self.sub_datas = \
            [replay_buffer.ReplayData() for _ in range(num_envs)]
        
        for index in range(num_envs):
            process = mp.Process(target=self.run, args=(index, env_prams))
            process.start()
            # self.env_conns[index].close()

    def run(self, index:int, prams:params.EnvParams):
        """子进程循环"""
        logger.info(f"start sub process: {index}")
        sub_env = SubEnv(self.Env(), prams, index)
        while True:
            request, action = self.sub2main[index].recv()
            if request == "step":
                self.sub2main[index].send(sub_env.step(action, False))
            elif request == "reset":
                self.sub2main[index].send(sub_env.reset())
            else:
                raise NotImplementedError

    def reset(self):
        """
        将其管理的所有子环境全部重置
        
        """
        request = "reset"
        action = None
        for pipe in self.main2sub:
            pipe.send((request, action))
        self.states = [agent_conn.recv()[0] for agent_conn in self.main2sub]
        return self.states

    
    def step(self, agent:ppo.Agent):
        """给所有子环境动作, 并记录每个子环境的`replay_data`"""
        apv = [agent.desision(s) for s in self.states]
        for pipe, rl_return in zip(self.main2sub, apv):
            pipe.send(("step", rl_return[0]))

        index = 0
        for rl_return, replay_data, pipe in zip(apv, self.sub_datas, self.main2sub):
            subenv_renturn = pipe.recv()  # [s, r, d, dbug] or None
            if subenv_renturn is not None:
                s1, r, d, _ = subenv_renturn
                a, p, v = rl_return
                replay_data.pack_step_data(self.states[index], a, p, r, v, d)
                if d:
                    self.replay_buffer.pack_episode_data(replay_data)
                self.states[index] = s1  # update states
            index += 1


    def get_states(self)->list[np.ndarray]:
        """获取所有子环境中的`state`"""
        return self.states

    def get_buffer(self):
        # 将所有子环境的数据汇总
        # for replay_data in self.sub_datas:
        #     self.replay_buffer.pack_episode_data(replay_data)
        return self.replay_buffer
    
    def clear(self):
        """清空`repaly_buffer`"""
        # for replay_data in self.sub_datas:
        #     replay_data.clear()
        self.replay_buffer.clear()



if __name__ == "__main__":
    pass