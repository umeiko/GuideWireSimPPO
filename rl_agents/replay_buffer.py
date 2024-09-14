import numpy as np
import logging

logger = logging.getLogger(__name__)

class ReplayData:
    """单幕交互中的数据,用来计算和记录优势值,每幕完成后需要清空"""
    def __init__(self) -> None:
        self.datas = {
                "s":[],
                "a":[],
                "p":[],
                "r":[],
                "v":[],
                "d":[],
            }
        
    def pack_step_data(self, state, action, prob, reward, value, done):
        """将单次交互的结果存储
        `state` `action` `prob` `reward` `value` `done`
        """
        self.datas["s"].append(state)
        self.datas["a"].append(action)
        self.datas["p"].append(prob)
        self.datas["r"].append(reward)
        self.datas["v"].append(value)
        self.datas["d"].append(done)

    def calc_gae(self, lambda_:float, gamma:float):
        '''在单局结束时计算GAE优势,返回gae收益及gae优势'''
        r = self.datas["r"]
        d = self.datas["d"]
        v = self.datas["v"]
        next_value = 0.
        rewards = []
        advantages = []
        gae = 0
        for reward, done, value in list(zip(r, d, v))[::-1]:	
            gae = gae * lambda_ * gamma
            gae += reward + gamma * next_value * (1. - done) - value  # <-- 这个是gae优势值
            next_value = value
            advantages.insert(0, gae)       # <-- 这个是gae优势值
            rewards.insert(0, gae + value)   # <-- 这里储存的是折算后总收益，没有减去基线的
        return rewards, advantages
    
    def clear(self):
        self.datas = {
            "s":[],
            "a":[],
            "p":[],
            "r":[],
            "v":[],
            "d":[],
        }
    

class ExperimentReplayBuffer():
    def __init__(self,) -> None:
        self.states  = []
        self.actions = []
        self.rewards = []
        self.values  = []
        self.gaes    = []
        self.old_probs  = []
        self.advantages = []

    def pack_episode_data(self, replaydata: ReplayData, lambda_=0.95, gamma=0.99):
        """将单幕的数据打包并传入池中, 将单幕存储器清空"""
        g, adv = replaydata.calc_gae(lambda_, gamma)
        self.states += replaydata.datas["s"]
        self.actions += replaydata.datas["a"]
        self.rewards += replaydata.datas["r"]
        self.values  += replaydata.datas["v"]
        self.old_probs += replaydata.datas["p"]
        self.gaes       += g
        self.advantages += adv
        # logging.info(f"r:{self.rewards}")
        # logging.info(f'd:{replaydata.datas["d"]}')
        # logging.info(f"g:{self.gaes}")
        # logging.info(f"v:{self.advantages}")
        replaydata.clear()

    def get_needed_data(self):
        """返回PPO训练所需的所有数据: `states` `actions` `rewards` `probs` `values` `gaes` `advantages`"""
        out_states = np.zeros((len(self.states), *self.states[0].shape), dtype=np.float32)
        for k, state in enumerate(self.states):
            out_states[k] = state

        return out_states, np.array(self.actions).astype(np.int32), np.array(self.rewards).astype(np.float32), \
                np.array(self.old_probs).astype(np.float32), np.array(self.values).astype(np.float32),\
                np.array(self.gaes).astype(np.float32), np.array(self.advantages).astype(np.float32)

    def clear(self):
        """清空回放池"""
        self.states  = []
        self.actions = []
        self.rewards = []
        self.values  = []
        self.gaes    = []
        self.old_probs  = []
        self.advantages = []
    
    def __len__(self):
        return len(self.states)