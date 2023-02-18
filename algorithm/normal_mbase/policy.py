from algorithm.normal_mbase.model import EnvNet,PolicyNet,ActorCritic
from models.preprocess import PreProcess
from utils.self_play import SelfPlay
import torch
import torch.nn as NN
import numpy as np
class policy:
    def __init__(self,config) -> None:
        self.EnvNet = EnvNet(config)
        self.agent = ActorCritic(config)
        self.config = config.normal_mbase
        self.backbone = PreProcess('resnet34')
        self.env_lossfun = NN.SmoothL1Loss()
        self.self_play = SelfPlay(config)

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

            select_indexes = np.random.randint(1,config.max_moves,
                (self.config.policynet_train_sample_num,
                    self.config.policy_train_batchsize/self.config.policynet_train_sample_num))
            
            # get batch via random choice
            # TODO : check size of batch 
            actions_batch = torch.Tensor([sampled_games[ii].actions_history[jj] 
                for ii in range(self.config.policynet_train_sample_num)
                    for jj in select_indexes[ii] ])
            actions_batch = NN.functional.one_hot(actions_batch,num_classes = 3)

            last_states_batch = torch.Tensor([sampled_games[ii].observations_history['rgb'][jj-1]
                for ii in range(self.config.policynet_train_sample_num)
                    for jj in select_indexes[ii]])
            current_states_batch = torch.Tensor([sampled_games[ii].observations_history['rgb'][jj]
                for ii in range(self.config.policynet_train_sample_num)
                    for jj in select_indexes[ii]])
            
            reward_batch = torch.Tensor([sampled_games[ii].reward_history[jj]
                for ii in range(self.config.policynet_train_sample_num)
                    for jj in select_indexes[ii]])

            self.agent.update((current_states_batch,actions_batch,reward_batch,last_states_batch))
    
    def train(self,replay_buffer,shared_storage):
        self.self_play.continuous_self_play(shared_storage,replay_buffer,self.agent)

        # train environment & policy by turns
        shared_storage.set_info('envpolicy_train_epoch' , 0)
        while shared_storage.get_info('envpolicy_train_epoch') < self.config.envpolicy_train_epoches:
            shared_storage.set_info('env_train_epoch' , 0)
            
            # train environment 
            while shared_storage.get_info('env_train_epoch') < self.config.env_train_epoches:
                self.train_env(replay_buffer,shared_storage)
                shared_storage.set_info('env_train_epoch' 
                    ,shared_storage.get_info('env_train_epoch') + 1)

            # collect new game history
            self.self_play.continuous_self_play(shared_storage,replay_buffer,self.agent)
            
            # train policy
            shared_storage.set_info('policy_train_epoch' , 0)
            while shared_storage.get_info('policy_train_epoch') < self.config.policy_train_epoches:
                self.train_policy(replay_buffer,shared_storage)
                shared_storage.set_info('policy_train_epoch'
                    ,shared_storage.get_info('policy_train_epoch') + 1)
            
            shared_storage.set_info('envpolicy_train_epoch' 
                ,shared_storage.get_info('envpolicy_train_epoch') + 1)


        
            
            
            


            


        
        
        