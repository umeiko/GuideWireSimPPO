import cv2
from env import env as _env
import params
from rl_agents import ppo, mult_processor
import numpy as np
import logging


def img_resize(img:np.ndarray, shape:tuple):
	# cv2的 x, y 坐标与 np 的相反
	img = cv2.resize(img, (shape[1], shape[0]))
	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	img = img.reshape(1, shape[0], shape[1])
	return 2. * img / 255 - 1


def eval(agent:ppo.Agent, Env:_env.BaseEnv, 
			env_param:params.EnvParams, render=True, eval_times=3):
	"""运行一局并测试性能"""
	# if not isinstance(Env, _env.BaseEnv):
	env = Env
	# else:
	# 	env = Env
	if render:
		w = mult_processor.SubRenderWindow(env_param)
	
	env.set_params(env_param)
	s = env.reset()
	avg_step = 0
	avg_reward = 0
	agent.ac_model.eval()
	for i in range(eval_times):
		for idx in range(len(env)):
			reward_total = 0
			spend_steps  = 0
			env.reset(idx)
			for _ in range(env_param.max_steps):
				if render:
					w.show(env.render())
				spend_steps += 1
				a, _, _ = agent.desision(s)
				s1, r, d, _ = env.step(a)
				s = s1
				reward_total += r
				if d:
					break
			if render:
				w.close()
			logging.info(f"eval {i}: {env.get_now_info()} -- reward: {reward_total}, steps: {spend_steps}")
		
			avg_reward += reward_total
			avg_step += spend_steps
		# logging.info(f"avg_reward: {avg_reward}, avg_steps: {avg_step}")
	
	logging.info(f"avg_step: {avg_step} / {eval_times * len(env)}")
	avg_reward /= (eval_times * len(env))
	avg_step /= (eval_times * len(env))
	
	logging.info(f"eval done: reward: {avg_reward}, steps: {avg_step}")
	return avg_reward, avg_step, s

def set_param(agent, param):
    # 读取参数
    for key in dir(param):
        if not callable(getattr(param, key)) and not key.startswith('__'):
            setattr(agent, key, getattr(param, key))

class Weight_Manager():
	pass