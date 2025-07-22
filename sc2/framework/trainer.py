"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import copy
import math
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.distributions import Categorical
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR

from .utils import MultiStageAdaptiveLRScheduler

class TrainerConfig:
    # optimization parameters
    max_epochs = 1000
    batch_size = 128
    learning_rate = 5e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 0.5
    weight_decay = 0.1  # only applied on matmul weights
    # checkpoint settings
    num_workers = 0  # for DataLoader

    rtg_min = 5.0  # 最小RTG值
    rtg_max = 25.0  # 最大RTG值
    rtg_momentum = 0.95  # RTG平滑因子
    win_rate_threshold = 0.6  # 胜率阈值
    return_scale = 0.2  # return影响因子
    use_lr_scheduler = False  # 是否使用学习率调度器

     # KL散度约束
    kl_coef = 0.001  # KL散度损失系数

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
class Trainer:

    def __init__(self, model, critic_model, config):
        self.model = model
        self.critic_model = critic_model
        self.config = config

        self.frozen_model = None  # 冻结模型
        # self.current_rtg = config.rtg_min
        # self.running_win_rate = 0.0
        # self.running_return = 0.0
    
        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        self.raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = self.raw_model.configure_optimizers(config, config.learning_rate)

        self.raw_critic_model = self.critic_model.module if hasattr(self.critic_model, "module") else self.critic_model
        self.critic_optimizer = self.raw_critic_model.configure_optimizers(config, config.learning_rate * 10)
         
        #add lr scheduler
        if self.config.use_lr_scheduler: 
            self.scheduler = MultiStageAdaptiveLRScheduler(
                self.optimizer,
                win_rate_threshold=0.6,
                win_rate_window=50,
                lr_bounds=(1e-5, 5e-5),
                warmup_epochs=5,
                exploration_epochs=20,
                restart_period=100,
                restart_factor=1.5
        )
        
            self.critic_scheduler = MultiStageAdaptiveLRScheduler(
                self.critic_optimizer,
                win_rate_threshold=0.6,
                win_rate_window=50,
                lr_bounds=(1e-4, 5e-4),  # Critic使用更大的学习率范围
                warmup_epochs=5,
                exploration_epochs=20,
                restart_period=100,
                restart_factor=1.5
        )

      
    
    def update_rtg(self, win_rate, episode_return):
        """动态更新RTG目标值"""
        # 更新运行时统计
        self.running_win_rate = self.config.rtg_momentum * self.running_win_rate + \
                               (1 - self.config.rtg_momentum) * win_rate
        self.running_return = self.config.rtg_momentum * self.running_return + \
                            (1 - self.config.rtg_momentum) * episode_return
        
        # 根据胜率和return计算目标RTG
        win_factor = min(1.0, self.running_win_rate / self.config.win_rate_threshold)
        return_factor = min(1.0, self.running_return * self.config.return_scale)
        
        # 综合因子
        target_factor = (win_factor + return_factor) / 2
        
        # 平滑更新RTG
        target_rtg = self.config.rtg_min + (self.config.rtg_max - self.config.rtg_min) * target_factor
        self.current_rtg = self.config.rtg_momentum * self.current_rtg + \
                          (1 - self.config.rtg_momentum) * target_rtg
        
        return self.current_rtg
    def set_frozen_model(self, frozen_model):
        #"""设置冻结模型"""
            self.frozen_model = frozen_model
    def train(self, dataset, train_critic=True):
        model, critic_model, config = self.raw_model, self.raw_critic_model, self.config
        target_model = copy.deepcopy(model)
        target_model.train(False)
        
        def run_epoch():
            model.train(True)
            critic_model.train(True)
            if self.config.mode == "offline":
                loader = DataLoader(dataset, shuffle=True, pin_memory=True, drop_last=True,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers)
            elif self.config.mode == "online":
                loader = DataLoader(dataset, shuffle=True, pin_memory=True, drop_last=True,
                                    batch_size=dataset.__len__(),
                                    num_workers=config.num_workers)
            else:
                raise NotImplementedError

            loss_info = 0
            kl_loss_info = 0
            pbar = tqdm(enumerate(loader), total=len(loader))

            # todo: check these inputs
            for it, (s, o, a, r, ava, v, rtg, ret, adv, t, pre_a, next_s, next_rtg, done) in pbar:
                # place data on the correct device
                s = s.to(self.device)
                o = o.to(self.device)
                a = a.to(self.device)
                r = r.to(self.device)
                ava = ava.to(self.device)
                v = v.to(self.device)
                rtg = rtg.to(self.device)
                ret = ret.to(self.device)
                adv = adv.to(self.device)
                t = t.to(self.device)
                pre_a = pre_a.to(self.device)
                next_s = next_s.to(self.device)
                next_rtg = next_rtg.to(self.device)
                done = done.to(self.device)

                # update actor
                with torch.set_grad_enabled(True):
                    logits = model(o, pre_a, rtg, t)
                     # 计算KL散度约束
                    kl_loss = torch.tensor(0.0, device=logits.device)
                    if self.frozen_model is not None and self.config.mode == "online":
                        with torch.no_grad():
                            frozen_logits = self.frozen_model(o, pre_a, rtg, t)
                        
                        # 屏蔽不可用动作
                        logits_masked = logits.clone()
                        frozen_logits_masked = frozen_logits.clone()
                        logits_masked[ava == 0] = -1e10
                        frozen_logits_masked[ava == 0] = -1e10
                        
                        # 计算KL散度
                        kl_loss = F.kl_div(
                            F.log_softmax(logits_masked, dim=-1),
                            F.softmax(frozen_logits_masked, dim=-1),
                            reduction='batchmean'
                        )
                        kl_loss_info = kl_loss.item()
                    if self.config.mode == "offline":
                        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), a.reshape(-1))
                        entropy_info = 0.
                        ratio_info = 0.
                        confidence_info = 0.
                    elif self.config.mode == "online":
                        adv = adv.reshape(-1, adv.size(-1))

                        logits[ava == 0] = -1e10
                        distri = Categorical(logits=logits.reshape(-1, logits.size(-1)))
                        target_a = a.reshape(-1)
                        log_a = distri.log_prob(target_a).unsqueeze(-1)

                        old_logits = target_model(o, pre_a, rtg, t).detach()
                        old_logits[ava == 0] = -1e10
                        old_distri = Categorical(logits=old_logits.reshape(-1, old_logits.size(-1)))
                        old_log_a = old_distri.log_prob(target_a).unsqueeze(-1)

                        imp_weights = torch.exp(log_a - old_log_a)
                        actor_loss_ori = imp_weights * adv
                        actor_loss_clip = torch.clamp(imp_weights, 1.0 - 0.2, 1.0 + 0.2) * adv
                        actor_loss = -torch.min(actor_loss_ori, actor_loss_clip)
                        # actor_loss = -log_a * adv

                        act_entropy = distri.entropy().unsqueeze(-1)
                        loss = actor_loss - 0.01 * act_entropy
                        # loss = actor_loss
                        # 添加KL散度约束
                        loss += config.kl_coef * kl_loss

                        entropy_info = act_entropy.mean().item()
                        ratio_info = imp_weights.mean().item()
                        confidence_info = torch.exp(log_a).mean().item()
                    else:
                        raise NotImplementedError
                    loss = loss.mean()
                    loss_info = loss.item()

                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()

                # update critic
                critic_loss_info = 0.
                if train_critic:
                    with torch.set_grad_enabled(True):
                        v_value = critic_model(s, pre_a, rtg, t)
                        v_clip = v + (v_value - v).clamp(-0.2, 0.2)
                        critic_loss_ori = F.smooth_l1_loss(v_value.view(-1, 1), ret.view(-1, 1), beta=10)
                        critic_loss_clip = F.smooth_l1_loss(v_clip.view(-1, 1), ret.view(-1, 1), beta=10)
                        critic_loss = torch.max(critic_loss_ori, critic_loss_clip)

                        critic_loss_info = critic_loss.mean().item()

                    critic_model.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic_model.parameters(), config.grad_norm_clip)
                    self.critic_optimizer.step()
                # if hasattr(self, 'scheduler'):
                #     # 更新胜率统计
                #     if done.any():
                #         n_threads = done.shape[0]  # 获取线程数
                #         wins = 0
                #         completed_episodes = 0
                        
                #         # 分别统计每个线程的胜负
                #         for thread_idx in range(n_threads):
                #             if done[thread_idx]:  # 当前线程的episode是否结束
                #                 completed_episodes += 1
                #                 # 注意：infos的结构是[thread_idx][0]['won']
                #                 if infos[thread_idx][0]['won']:
                #                     wins += 1
                        
                #         # 只在有完成的episode时更新胜率
                #         if completed_episodes > 0:
                #             thread_win_rate = wins / completed_episodes
                #             self.scheduler.update_win_rate(thread_win_rate)
                    
                    # 更新学习率
                    # self.scheduler.step()
                    # if train_critic:
                    #     self.critic_scheduler.step()    
                
                # report progress
                pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}.")
            return loss_info, critic_loss_info, entropy_info, ratio_info, confidence_info

        actor_loss_ret, critic_loss_ret, entropy, ratio, confidence = 0., 0., 0., 0., 0.
        for epoch in range(config.max_epochs):
            actor_loss_ret, critic_loss_ret, entropy, ratio, confidence = run_epoch()
            
         

        return actor_loss_ret, critic_loss_ret, entropy, ratio, confidence
    def update_scheduler(self, win_rate, sample_return):
        """基于性能指标直接更新学习率
        Args:
            win_rate: 当前episode的胜率
            sample_return: 当前episode的平均回报
        """
        if not hasattr(self, 'scheduler'):
            return
            
        try:
            # 更新调度器的胜率统计
            self.scheduler.update_win_rate(win_rate)
            self.scheduler.step()
            
            if hasattr(self, 'critic_scheduler'):
                self.critic_scheduler.step()
                
            # 打印当前状态
            print(f"\n=== 学习率更新 ===")
            print(f"当前胜率: {win_rate:.3f}")
            print(f"当前回报: {sample_return:.3f}")
            print(f"Actor学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            if hasattr(self, 'critic_optimizer'):
                print(f"Critic学习率: {self.critic_optimizer.param_groups[0]['lr']:.6f}")
            print("===============\n")
                
        except Exception as e:
            print(f"更新调度器时出错: {str(e)}")