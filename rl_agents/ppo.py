import sys 
sys.path.append("..") 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from  rl_agents.replay_buffer import *
import logging
import os
import time


logger = logging.getLogger(__name__)




OPT_TYPE = {
    "adam":torch.optim.Adam,
    "sgd":torch.optim.SGD,
}

class BaseNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.epoch:torch.Tensor = torch.tensor([0], dtype=torch.int32)
        # self.fc_epo = nn.Linear(1,1)
        # self.fc_epo.bias.data.fill_(0.0)

    ...

class Agent():
    def __init__(self, ac_model:BaseNetwork,
                    lr = 2e-4,          # 学习率
                    p_c = 1.0,          # Critic学习参数
                    gamma = 0.98,       # 奖励衰减系数
                    lambda_ = 1.0,      # GAE系数
                    beta = 0.01,        # 熵损失系数
                    epsilon = 0.2,      # PPO裁剪系数
                    batch_size = 16,    # 训练批量
                    exp_reuse_rate = 10,   # 经验回放复用率
                    device = None
                ) -> None:
        if device is None:
            self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.ac_model = ac_model
        self.lr = lr
        self.c_coef = p_c
        self.gamma = gamma
        self.lambda_ = lambda_
        self.beta = beta
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.exp_reuse_rate = exp_reuse_rate
        self.epoch = 0
        self.optm:torch.optim.Optimizer = None
        self.optim_type = "adam"
    
    def plot_attrs(self):
        print("Agent's attributes:")
        print("----------------------")
        for key in dir(self):
            attr = getattr(self, key)
            if not callable(attr) and not key.startswith('__'):
                print(key, ':\t\t', attr)
        print("----------------------")

    def load(self, path):
        try:
            checkpoint = torch.load(path)
            self.epoch = checkpoint['epoch']
            self.ac_model.load_state_dict(checkpoint['model_state_dict'])
            type_opt = checkpoint['type_opt']
            self.optm:torch.optim.Optimizer = OPT_TYPE[type_opt](self.ac_model.parameters())
            self.optm.load_state_dict(checkpoint['optimizer_state_dict'])
            return True
        
        except BaseException as e:
            logging.error(f"loading {path} with :{e}")
            return False
        
    
    def save(self, epo:int, path):  
        if isinstance(self.optm, torch.optim.Adam):
            type_opt = "adam"
        elif isinstance(self.optm, torch.optim.SGD):
            type_opt = "sgd"
        else:
            type_opt = "sgd"
        try:
            torch.save({
                'epoch': epo,
                'type_opt': type_opt,
                'model_state_dict': self.ac_model.state_dict(),
                'optimizer_state_dict': self.optm.state_dict(),
                }, path)
            return True
        except BaseException as e:
            logging.error(f"saving {path} with :{e}")
            return False



    def desision(self, state:torch.Tensor):
        '''输入【一个】状态并获得决策，决策概率及状态价值。'''
        state:torch.Tensor = torch.Tensor(state).unsqueeze(0).to(torch.float32).to(self.device)
        predict_prob, value = self.ac_model(state)
        action = torch.multinomial(predict_prob, 1, False)[0]
        predict_prob: torch.Tensor = predict_prob.squeeze()
        value: torch.Tensor = value.squeeze()
        return int(action), float(predict_prob.squeeze()[int(action)]), float(value)


    def forward_fn(self, s_batch:torch.Tensor, a_batch:torch.Tensor, 
                   gae_batch:torch.Tensor, advantages_batch:torch.Tensor, 
                   old_probs_batch:torch.Tensor):
        new_probs, values = self.ac_model(s_batch)
        # π(At|St, θ) / π_old(At|St, θ)
        ratio = torch.gather(new_probs, 1, a_batch)
        ratio = ratio / old_probs_batch

        surr1 = ratio * advantages_batch
        # 通过裁剪 π(At|St, θ) / π_old(At|St, θ) 到1的附近，限制过大的梯度更新，来自PPO论文
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages_batch
        # 更新较小的那一个梯度
        actor_loss = - torch.mean(torch.minimum(surr1, surr2))
        # 基线价值函数的损失函数，希望与GAE收益较为接近
        critic_loss = self.c_coef * F.smooth_l1_loss(gae_batch, values, reduction="mean")
        # 熵损失的计算公式 
        # 熵损失比较大，则鼓励智能体保持足够的探索
        # 熵损失比较小，则鼓励智能体输出结果更加确定
        entropy = torch.mean(torch.sum(-new_probs * torch.log(torch.clamp(new_probs, min=1e-5)), axis=1))
        entropy_loss = -self.beta * entropy
        final_loss = actor_loss + critic_loss + entropy_loss
        return final_loss, (actor_loss, critic_loss, entropy_loss)


    def learn(self, replaybuffer:ExperimentReplayBuffer)->list[float, float, float, float]:
        """训练并更新模型参数"""
        if self.optm is None:
            self.optm = OPT_TYPE[self.optim_type](self.ac_model.parameters(), lr=self.lr)
            # self.optm = torch.optim.Adam(self.ac_model.parameters(), lr=self.lr)      # 构建优化器
            logging.info(f"building optim:{self.optm}")

        s, a, r, old_probs, value, gae, advantages,  = replaybuffer.get_needed_data()
        avg_losses = [0,0,0,0]  # 将此次计算得到的平均损失输出用于记录
        times = round(self.exp_reuse_rate * len(s) / self.batch_size)
        self.ac_model.train()
        start_time = time.time()

        for _ in range(times):
            indice = torch.randperm(len(s))[:self.batch_size]  # 随机采样一部分
            # indice = list(randperm(len(s))[:self.batch_size])  # 随机采样一部分
            s_batch = torch.tensor(s[indice]).to(torch.float32).to(self.device) 
            a_batch = torch.tensor(a[indice]).unsqueeze(1).to(torch.int64).to(self.device) 
            gae_batch =  torch.tensor(gae[indice]).to(torch.float32).unsqueeze(-1).to(self.device) 
            advantages_batch =  torch.tensor(advantages[indice]).unsqueeze(-1).to(torch.float32).to(self.device)  
            old_probs_batch =  torch.tensor(old_probs[indice]).unsqueeze(-1).to(torch.float32).to(self.device)  

            loss, debug_msg = self.forward_fn(s_batch, a_batch, gae_batch, advantages_batch, old_probs_batch)
            actor_loss, critic_loss, entropy_loss = debug_msg
            self.optm.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac_model.parameters(), 0.5)
            self.optm.step()
            # 累加损失
            avg_losses[0] += float(actor_loss) / times
            avg_losses[1] += float(critic_loss) / times
            avg_losses[2] += float(entropy_loss) / times
            avg_losses[3] += float(loss) / times
        logging.info(f"Trained with {len(replaybuffer)} datas in buffer, cost {time.time()-start_time:.3f} s")
        return avg_losses


if __name__ == "__main__":
    ...
