from algorithm.normal_mbase.model import EnvNet,PolicyNet,ActorCritic
from models.preprocess import PreProcess
from utils.self_play import SelfPlay
import torch
import torch.nn as NN
import numpy as np
class policy:
    def __init__(self,config,writer) -> None:
        self.writer = writer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.EnvNet = EnvNet(config,self.writer).to(self.device)
        self.agent = ActorCritic(config,self.writer)
        self.orig_config = config
        self.config = config.normal_mbase # special config for model-based AC
        self.backbone = PreProcess('resnet34')
        self.env_lossfun = NN.SmoothL1Loss()
        self.self_play = SelfPlay(config,self.writer)

    def train_env(self,config,replay_buffer,shared_storage):
        while shared_storage.get_info('env_train_epoch') < self.config.envnet_train_epochs:
            sampled_games = replay_buffer.sample_games(
                self.config.envnet_train_sample_num)
            select_indexes = np.random.randint(1,config.play_steps,
                (self.config.envnet_train_sample_num,
                    int(self.config.env_train_batchsize/self.config.envnet_train_sample_num)))
            
            # get batch via random choice
            # TODO : check size of batch

            actions_batch = torch.tensor([sampled_games[ii][1].actions_history[jj] 
                for ii in range(self.config.envnet_train_sample_num)
                    for jj in select_indexes[ii] ]).to(dtype =torch.int64,device = self.device)

            actions_batch = NN.functional.one_hot(actions_batch,num_classes = 4)
            
            last_states_batch = torch.tensor(np.array([sampled_games[ii][1].observations_history['rgb'][jj-1]
                for ii in range(self.config.envnet_train_sample_num)
                    for jj in select_indexes[ii]])).to(dtype =torch.float32,device = self.device)
            current_states_batch = torch.tensor(np.array([sampled_games[ii][1].observations_history['rgb'][jj]
                for ii in range(self.config.envnet_train_sample_num)
                    for jj in select_indexes[ii]])).to(dtype =torch.float32,device = self.device)
            
            # get feature of state images
            last_states_batch = self.backbone.go(last_states_batch)
            current_states_batch = self.backbone.go(current_states_batch)

            self.EnvNet.train(actions_batch,last_states_batch,current_states_batch)
            shared_storage.set_info(
                'env_train_epoch',shared_storage.get_info('env_train_epoch') + 1)
                
     
    def train_policy(self,config,replay_buffer,shared_storage):
        shared_storage.set_info('policy_train_epoch', 0)
        while shared_storage.get_info('policy_train_epoch') < self.config.policynet_train_epochs:
            sampled_games = replay_buffer.sample_games(
                self.config.envnet_train_sample_num)
            select_indexes = np.random.randint(1,config.play_steps,
                (self.config.envnet_train_sample_num,
                    int(self.config.env_train_batchsize/self.config.envnet_train_sample_num)))
            
            # get batch via random choice
            # TODO : check size of batch
            # print(select_indexes)
            # if (select_indexes == [[1,6,7],[6,7 ,1], [1 ,5, 6], [4 ,2, 3], [2 ,2, 9], [2 ,6 ,8], [6 ,4 ,7],[4, 3, 4], [9 ,5 ,6],[5 ,9 ,8]]).all():
            #     breakpoint()
            actions_batch = torch.tensor([sampled_games[ii][1].actions_history[jj] 
                for ii in range(self.config.envnet_train_sample_num)
                    for jj in select_indexes[ii] ]).to(dtype =torch.int64,device = self.device)

            actions_batch = NN.functional.one_hot(actions_batch,num_classes = 4)
            
            last_states_batch = torch.tensor(np.array([sampled_games[ii][1].observations_history['rgb'][jj-1]
                for ii in range(self.config.envnet_train_sample_num)
                    for jj in select_indexes[ii]])).to(dtype =torch.float32,device = self.device)
            current_states_batch = torch.tensor(np.array([sampled_games[ii][1].observations_history['rgb'][jj]
                for ii in range(self.config.envnet_train_sample_num)
                    for jj in select_indexes[ii]])).to(dtype =torch.float32,device = self.device)
            
            # get feature of state images
            last_states_batch = self.backbone.go(last_states_batch)
            current_states_batch = self.backbone.go(current_states_batch)            
            rewards_batch = torch.tensor([sampled_games[ii][1].rewards_history[jj]
                for ii in range(self.config.policynet_train_sample_num)
                    for jj in select_indexes[ii]]).reshape([-1,1]).to(dtype =torch.float32,device = self.device)

            self.agent.update((current_states_batch,actions_batch,rewards_batch,last_states_batch))
            shared_storage.set_info('policy_train_epoch',
                shared_storage.get_info('policy_train_epoch') + 1)
            print(shared_storage.get_info('policy_train_epoch'))
    
    def train(self,replay_buffer,shared_storage):
        self.self_play.continuous_self_play(shared_storage,replay_buffer,self.agent)

        # train environment & policy by turns
        shared_storage.set_info('envpolicy_train_epoch' , 0)
        while shared_storage.get_info('envpolicy_train_epoch') < self.config.envpolicy_train_epoches:
            shared_storage.set_info('env_train_epoch' , 0)
            
            # train environment 
            self.train_env(self.orig_config,replay_buffer,shared_storage)


            # collect new game history
            self.self_play.continuous_self_play(shared_storage,replay_buffer,self.agent)
            
            # train policy
            self.train_policy(self.orig_config,replay_buffer,shared_storage)
            
            shared_storage.set_info('envpolicy_train_epoch' 
                ,shared_storage.get_info('envpolicy_train_epoch') + 1)


        
            
            
            


            


        
        
        