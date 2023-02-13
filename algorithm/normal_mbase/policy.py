from algorithm.normal_mbase.model import EnvNet,PolicyNet
from models.preprocess import PreProcess
import torch
import torch.nn as NN
import numpy as np
class policy:
    def __init__(self,config) -> None:
        self.EnvNet = EnvNet(config)
        self.PolicyNet = PolicyNet(config)
        self.config = config.normal_mbase
        self.backbone = PreProcess('resnet34')
        self.env_lossfun = NN.SmoothL1Loss()
        self.env_optimizer = torch.optim.Adam()

    def train_env(self,config,replay_buffer,shared_storage):
        while shared_storage.get_info('env_train_epoch') < self.config.envnet_train_epochs:
            sampled_games = replay_buffer.sample_games(
                self.config.envnet_train_sample_num)
            
            select_indexes = np.random.randint(1,config.max_moves,
                (self.config.envnet_train_sample_num,
                    self.config.env_train_batchsize/self.config.envnet_train_sample_num))
            
            # get batch via random choice
            # TODO : check size of batch 
            actions_batch = torch.Tensor([sampled_games[ii].actions_history[jj] 
                for ii in range(self.config.envnet_train_sample_num)
                    for jj in select_indexes[ii] ])
            actions_batch = NN.functional.one_hot(actions_batch,num_classes = 3)
            last_states_batch = torch.Tensor([sampled_games[ii].observations_history['rgb'][jj-1]
                for ii in range(self.config.envnet_train_sample_num)
                    for jj in select_indexes[ii]])
            current_states_batch = torch.Tensor([sampled_games[ii].observations_history['rgb'][jj]
                for ii in range(self.config.envnet_train_sample_num)
                    for jj in select_indexes[ii]])
            
            # get feature of state images
            last_states_batch = self.backbone.go(last_states_batch)
            current_states_batch = self.backbone.go(current_states_batch)

            self.EnvNet.train(actions_batch,last_states_batch,current_states_batch)
            shared_storage.set_info(
                'env_train_epoch',shared_storage.get_info('env_train_epoch') + 1)
                
    
    def train_policy(self,config,replay_buffer,shared_storage):
        while shared_storage.get_info('policy_train_epoch') < self.config.policynet_train_epochs:
            sampled_games = replay_buffer.sample_games(
                self.config.policynet_train_sample_num)

            rewards_batch = [sampled_games[ii].rewards_history 
                for ii in range(self.config.policynet_train_sample_num)]

            sumed_rewards = []
            for rewards_history in rewards_batch:
                sum = 0
                for index,reward in enumerate(rewards_batch):
                    sum += reward * (config.discount_rate ** index)
                sumed_rewards.append(sum)
            sumed_rewards
            
            


            


        
        
        