import sys
# print(sys.path)

from rl_agents import ppo, mult_processor
from models import example
import logging
import params
import torch
import os
import utils
from tqdm import tqdm
import shutil
from torch.utils.tensorboard import SummaryWriter
import time
import traceback
import argparse


parser = argparse.ArgumentParser(description="Selecting mode.")
# 添加命令行参数
parser.add_argument("--mode", type=str, default="train", help="train or test")
# 解析命令行参数
ARGS = parser.parse_args()
if ARGS.mode not in ["train", "test"]:
    raise NotImplementedError(f"Illigel arg '{ARGS.mode}' ")


# 日志模块初始化
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logger = logging.getLogger(__name__)
current_time = time.strftime("%m-%d_%H-%M", time.localtime())


# 全局定义训练环境
from env import cartpole
ENV      = cartpole.CartPoleEnv
ENV_NAME = "cartpole"


def main():
    logger.info('Started')
    train_param = params.TrainParams()
    env_param   = params.EnvParams()
    run_param    = params.RuntimeParams()

    train_param.load_from_json(f'./env/{ENV_NAME}')
    env_param.load_from_json(f'./env/{ENV_NAME}')
    run_param.load_from_json(f'./env/{ENV_NAME}')
    
    logging.basicConfig(filename=f'./logs/{current_time}_{run_param.task_name}.log',
                         level=logging.INFO,format=log_format)
    logging.info(f'train_param:\n{train_param.get_msg()}')
    logging.info(f'envir_param:\n{env_param.get_msg()}')
    logging.info(f'run_param:\n{run_param.get_msg()}')


    model = example.example_network(env_param.input_dense, env_param.actions)
    model = model.to(train_param.device)
    agent = ppo.Agent(model)
    weight_path = os.path.join("./weights", run_param.task_name)
    os.makedirs(weight_path, exist_ok=True)
    
    # 读取参数
    for key in dir(train_param):
        if not callable(getattr(train_param, key)) and not key.startswith('__'):
            setattr(agent, key, getattr(train_param, key))

    writer = SummaryWriter(log_dir=f'./logs/{run_param.task_name}')
    
    # 加载权重
    last_epo = 0
    if agent.load(os.path.join(weight_path, f"last.pth")):
        last_epo = int(agent.epoch)
        logging.info(f'loaded weight in {os.path.join(weight_path, f"last.pth")} with {last_epo} epoches')
    else:
        logging.info(f'create new weights in {os.path.join(weight_path, f"last.pth")}')

    if ARGS.mode == 'train':
        mp_manager = mult_processor.MultEnvStepper(ENV, env_param, run_param)
        mp_manager.reset()
    elif ARGS.mode == 'test':
        mp_manager = None
        run_param.save_interval = train_param.num_epochs+1
        run_param.plot_interval = 1
        now_epo = 0
        logging.info(f'Entering test mode...')

    eval_env = ENV()
    env_param.apply(eval_env)

    # 主循环
    losses = None
    best = 0
    for epo in tqdm(range(train_param.num_epochs)):
        now_epo = epo + last_epo 

        # 训练过程
        if ARGS.mode == 'train':
            last_time = time.time()
            for _ in range(env_param.max_steps):			
                mp_manager.step(agent)
            
            buffer = mp_manager.get_buffer()
            logging.info(f'Epoch {now_epo}: stepcost {time.time()-last_time:.2g} s')

            # 训练神经网络
            if len(buffer) >= train_param.batch_size:
                losses = agent.learn(buffer)
                mp_manager.clear()
            
            # tensorboard
            if losses is not None:
                writer.add_scalar('Loss/actor', losses[0], now_epo)
                writer.add_scalar('Loss/critic', losses[1], now_epo)
                writer.add_scalar('Loss/entropy', losses[2], now_epo)
                writer.add_scalar('Loss/all', losses[3], now_epo)

        # 验证过程
        if ((epo % run_param.plot_interval == 0)):
            total, steps, _ = utils.eval(agent, eval_env, env_param, False)
            writer.add_scalar('Reward/reward', total, now_epo)
            writer.add_scalar('Reward/Spend Steps', steps, now_epo)
            # writer.add_image("Last_result Image", img.reshape(env_param.image_size), 
            #     global_step=epo, dataformats="HW",)
            plotloss = 0.00 if losses is None else losses[0]
            print(f"# of episode :{now_epo}, score : {total}, steps :{steps}, loss: {plotloss:.2f}")
            logging.info(f"# of episode :{now_epo}, score : {total}, steps :{steps}, loss: {plotloss:.2f}")

        # 权重存储
        if ((epo % run_param.save_interval == 0)):
            plotloss = 0.00 if losses is None else losses[0]
            total, steps, _ = utils.eval(agent, eval_env, env_param, False)
            os.makedirs(weight_path, exist_ok=True)
            if total > best:
                agent.save(now_epo, os.path.join(weight_path, f"best.pth"))
                logging.info(f"# best saved: score : {total}, steps :{steps}, loss: {plotloss:.2f}")
                best = total
            agent.save(now_epo, os.path.join(weight_path, f"last.pth"))
            logging.info(f"# last saved: score : {total}, steps :{steps}, loss: {plotloss:.2f}")
            print("weights saved")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(traceback.format_exc())
