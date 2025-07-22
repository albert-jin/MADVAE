import setproctitle
import logging
import argparse
import torch
import sys
import os

from tensorboardX.writer import SummaryWriter
from framework.utils import set_seed
from framework.trainer import Trainer, TrainerConfig
from framework.utils import get_dim_from_space
from envs.env import Env
from framework.buffer import ReplayBuffer
from framework.rollout import RolloutWorker
from datetime import datetime, timedelta
from models.gpt_model import GPT, GPTConfig,GPT2
# from models.mlp_model import GPT, GPTConfig,GPT2
from models.convformer_model import GPT1,GPT1Config 
from framework.utils import CPUManager

from envs import config

# args = sys.argv[1:]
parser = argparse.ArgumentParser()






parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=1)
parser.add_argument('--model_type', type=str, default='state_only')#'state_only
parser.add_argument('--eval_episodes', type=int, default=32)#32
parser.add_argument('--max_timestep', type=int, default=400)
parser.add_argument('--log_dir', type=str, default='./logs/')
parser.add_argument('--save_log', type=bool, default=True)
parser.add_argument('--exp_name', type=str, default='easy_trans')
parser.add_argument('--pre_train_model_path', type=str, default='../offline_model/')

parser.add_argument('--offline_map_lists', type=list, default=['corridor','corridor','corridor','corridor','corridor'])#['3s_vs_4z', '2m_vs_1z', '3m', '2s_vs_1sc', '3s_vs_3z']
parser.add_argument('--offline_episode_num', type=list, default=[60,60,60,60,60])#[200, 200, 200, 200, 200]
parser.add_argument('--offline_data_quality', type=list, default=['good'])
parser.add_argument('--offline_data_dir', type=str, default='../path/to/offline/data/')#'../offline_data/'

parser.add_argument('--offline_epochs', type=int, default=10)
parser.add_argument('--offline_mini_batch_size', type=int, default=128)
parser.add_argument('--offline_lr', type=float, default=1e-4)#5e-4
parser.add_argument('--offline_eval_interval', type=int, default=1)
parser.add_argument('--offline_train_critic', type=bool, default=True)
parser.add_argument('--offline_model_save', type=bool, default=True)

parser.add_argument('--online_buffer_size', type=int, default=64)#64
parser.add_argument('--online_epochs', type=int, default=5000)
parser.add_argument('--online_ppo_epochs', type=int, default=10)#10
parser.add_argument('--online_lr', type=float, default=1e-4)#5e-4
parser.add_argument('--online_eval_interval', type=int, default=1)
parser.add_argument('--online_train_critic', type=bool, default=True)
parser.add_argument('--online_pre_train_model_load', type=bool, default=False)
parser.add_argument('--online_pre_train_model_id', type=int, default=9)



def get_env_dims(env):
    
    try:
        global_obs_dim = get_dim_from_space(env.real_env.share_observation_space)
        local_obs_dim = get_dim_from_space(env.real_env.observation_space)
        action_dim = get_dim_from_space(env.real_env.action_space)
        
        print(f"从环境获取维度: global={global_obs_dim}, local={local_obs_dim}, action={action_dim}")
        return global_obs_dim, local_obs_dim, action_dim
    except Exception as e:
        print(f"获取环境维度失败: {e}")
        return None
def setup_device(args):
    
    if not args.use_cpu and torch.cuda.is_available():
        # 设置可见的CUDA设备
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
        device = torch.device("cuda")
        print(f"使用GPU: {args.cuda_id}")
        print(f"当前GPU型号: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("使用CPU训练")
    return device

def setup_process_title(args):
    
    title = f"{args.exp_name}@"
    setproctitle.setproctitle(title)
    print(f"进程标题设置为: {title}")


# args = parser.parse_args(args, parser)
args = parser.parse_args()
set_seed(args.seed)
setup_process_title(args)
torch.set_num_threads(8)

    


cur_time = datetime.now() + timedelta(hours=0)
args.log_dir += cur_time.strftime("[%m-%d]%H.%M.%S")

writter = SummaryWriter(args.log_dir+args.exp_name) if args.save_log else None

eval_env = Env(args.eval_episodes)
online_train_env = Env(args.online_buffer_size)
global_obs_dim = 267
local_obs_dim = 252
action_dim = 15
block_size = args.context_length * 3

print("global_obs_dim: ", global_obs_dim)
print("local_obs_dim: ", local_obs_dim)
print("action_dim: ", action_dim)



mconf_actor = GPTConfig(local_obs_dim, action_dim, block_size,
                        n_layer=2, n_head=2, n_embd=32, model_type=args.model_type, max_timestep=args.max_timestep)#2 2 32
model = GPT(mconf_actor, model_type='actor')

mconf_critic = GPTConfig(global_obs_dim, action_dim, block_size,
                         n_layer=2, n_head=2, n_embd=32, model_type=args.model_type, max_timestep=args.max_timestep)#2 2 32
critic_model = GPT(mconf_critic, model_type='critic')

mconf_frozen_actor = GPTConfig(local_obs_dim, action_dim, block_size,
                              n_layer=2, n_head=2, n_embd=32, model_type=args.model_type, max_timestep=args.max_timestep)
frozen_actor_model = GPT(mconf_frozen_actor, model_type='actor')

# device = setup_device(args)
# model = model.to(device)
# critic_model = critic_model.to(device)

if torch.cuda.is_available():  #torch.cuda.is_available():
    device = torch.cuda.current_device()
    model = torch.nn.DataParallel(model).to(device)
    critic_model = torch.nn.DataParallel(critic_model).to(device)
    frozen_actor_model = torch.nn.DataParallel(frozen_actor_model).to(device)
buffer = ReplayBuffer(block_size, global_obs_dim, local_obs_dim, action_dim)

rollout_worker = RolloutWorker(model, critic_model,frozen_actor_model, buffer, global_obs_dim, local_obs_dim, action_dim)
rollout_worker.frozen_actor_model = frozen_actor_model
used_data_dir = []
for map_name in args.offline_map_lists:
    source_dir = args.offline_data_dir + map_name
    for quality in args.offline_data_quality:
        used_data_dir.append(f"{source_dir}/{quality}/")

buffer.load_offline_data(used_data_dir, args.offline_episode_num, max_epi_length=eval_env.max_timestep)
offline_dataset = buffer.sample()
offline_dataset.stats()

offline_tconf = TrainerConfig(max_epochs=1, batch_size=args.offline_mini_batch_size, learning_rate=args.offline_lr,
                              num_workers=0, mode="offline")
offline_trainer = Trainer(model, critic_model, offline_tconf)

# target_rtgs = offline_dataset.max_rtgs
target_rtgs = 20.
print("offline target_rtgs: ", target_rtgs)
for i in range(args.offline_epochs):
    offline_actor_loss, offline_critic_loss, _, __, ___ = offline_trainer.train(offline_dataset,
                                                                                args.offline_train_critic)
    if args.save_log:
        writter.add_scalar('offline/{args.map_name}/offline_actor_loss', offline_actor_loss, i)
        writter.add_scalar('offline/{args.map_name}/offline_critic_loss', offline_critic_loss, i)
    if i % args.offline_eval_interval == 0:
        aver_return, aver_win_rate, _ = rollout_worker.rollout(eval_env, target_rtgs, train=False)
        print("offline epoch: %s, return: %s, eval_win_rate: %s" % (i, aver_return, aver_win_rate))
        if args.save_log:
            writter.add_scalar('offline/{args.map_name}/aver_return', aver_return.item(), i)
            writter.add_scalar('offline/{args.map_name}/aver_win_rate', aver_win_rate, i)
    if args.offline_model_save and i==args.offline_epochs-1:
        actor_path = args.pre_train_model_path + args.exp_name + '/actor'
        if not os.path.exists(actor_path):
            os.makedirs(actor_path)
        critic_path = args.pre_train_model_path + args.exp_name + '/critic'
        if not os.path.exists(critic_path):
            os.makedirs(critic_path)
        torch.save(model.state_dict(), actor_path + os.sep + str(i) + '.pkl')
        torch.save(critic_model.state_dict(), critic_path + os.sep + str(i) + '.pkl')


if args.online_epochs > 0 and args.online_pre_train_model_load:
    actor_path = args.pre_train_model_path + args.exp_name + '/actor/' + str(args.online_pre_train_model_id) + '.pkl'
    critic_path = args.pre_train_model_path + args.exp_name + '/critic/' + str(args.online_pre_train_model_id) + '.pkl'
    model.load_state_dict(torch.load(actor_path), strict=False)
    critic_model.load_state_dict(torch.load(critic_path), strict=False)
   
    rollout_worker.frozen_actor_model.load_state_dict(torch.load(actor_path), strict=False)
    for param in rollout_worker.frozen_actor_model.parameters():
        param.requires_grad = False
    rollout_worker.frozen_actor_model.eval()

online_tconf = TrainerConfig(max_epochs=args.online_ppo_epochs, batch_size=0,
                             learning_rate=args.online_lr, num_workers=0, mode="online",use_lr_scheduler = True)
online_trainer = Trainer(model, critic_model, online_tconf)
online_trainer.set_frozen_model(frozen_actor_model)
buffer.reset(num_keep=0, buffer_size=args.online_buffer_size)

total_steps = 0
rollout_worker.trainer = online_trainer
for i in range(args.online_epochs):
    
    #sample_return, win_rate, steps = rollout_worker.rollout(online_train_env, target_rtgs, train=True) #  win_rate target_rtgs
    sample_return, win_rate, steps = rollout_worker.rollout(online_train_env, target_rtgs, train=True)
   
    
    total_steps += steps
    online_dataset = buffer.sample()
    online_actor_loss, online_critic_loss, entropy, ratio, confidence = online_trainer.train(online_dataset,
                                                                                             args.online_train_critic)
    if args.save_log:
        writter.add_scalar('online/{args.map_name}/online_actor_loss', online_actor_loss, total_steps)
        writter.add_scalar('online/{args.map_name}/online_critic_loss', online_critic_loss, total_steps)
        writter.add_scalar('online/{args.map_name}/entropy', entropy, total_steps)
        writter.add_scalar('online/{args.map_name}/ratio', ratio, total_steps)
        writter.add_scalar('online/{args.map_name}/confidence', confidence, total_steps)
        writter.add_scalar('online/{args.map_name}/sample_return', sample_return, total_steps)


    print("sample return: %s, online target_rtgs: %s" % (sample_return, target_rtgs))

    
    if i % args.online_eval_interval == 0:
        #aver_return, aver_win_rate, _ = rollout_worker.rollout(eval_env, target_rtgs, train=False)
        aver_return, aver_win_rate, _ = rollout_worker.rollout_with_imitate(eval_env, target_rtgs, train=False)
        print("online steps: %s, return: %s, eval_win_rate: %s" % (total_steps, aver_return, aver_win_rate))
        if args.save_log:
            writter.add_scalar('online/{args.map_name}/aver_return', aver_return.item(), total_steps)
            writter.add_scalar('online/{args.map_name}/aver_win_rate', aver_win_rate, total_steps)

online_train_env.real_env.close()
eval_env.real_env.close()
