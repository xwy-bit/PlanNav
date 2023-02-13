import torch
import torch.nn as NN
import torch.nn.functional as F

class EnvNet(NN.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.layer_num = config.normal_mbase.envnet_layer_num
        self.f1 = NN.Linear(self.layer_num[0],self.layer_num[1])
        self.f2 = NN.Linear(self.layer_num[1],self.layer_num[2])
        self.f3 = NN.Linear(self.layer_num[2],self.layer_num[3])
        self.f4 = NN.Linear(self.layer_num[3],self.layer_num[4],bias=False)
        self.Fun = F.gelu
        self.lossfun = NN.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.parameters(),lr = config.normal_mbase.envnet_train_lr)
    def forward(self,action,last_state):
        '''
        concat action & last_state
            action    : N * 1
            last_state: M * 1 
        '''

        hidden = torch.concat([action,last_state],1).reshape(-1)
        hidden = self.f1(hidden)
        hidden = self.Fun(hidden)
        hidden = self.f2(hidden)
        hidden = self.Fun(hidden)
        hidden = self.f3(hidden)
        hidden = self.Fun(hidden)
        state = self.f4(hidden)
        return state
    def train(self,action,last_state,target_state):
        predict_states = self.forward(action,last_state)
        self.optimizer.zero_grad()
        loss = self.lossfun(target_state,predict_states)
        loss.backward()
        self.optimizer.step()
class PolicyNet(NN.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.layer_num = config.normal_mbase.policy_layer_num
        self.f1 = NN.Linear(self.layer_num[0],self.layer_num[1])
        self.f2 = NN.Linear(self.layer_num[1],self.layer_num[2])
        self.f3 = NN.Linear(self.layer_num[2],self.layer_num[3])
        self.f4 = NN.Linear(self.layer_num[3],self.layer_num[4])
        self.Fun = F.gelu
        self.optimizer = torch.optim.Adam(
            self.parameters(),lr =config.normal_mbase.policy_train_lr)
    def forward(self,state):
        hidden = self.f1(state)
        hidden = self.Fun(hidden)
        hidden = self.f2(hidden)
        hidden = self.Fun(hidden)
        hidden = self.f3(hidden)
        hidden = self.Fun(hidden)
        action = self.f4(hidden)
        return action
    def train(self,rewards_batch:list):
        for rewards in rewards_batch:
           rewards = -rewards
           self.optimizer.zero_grad()
           rewards.backward()
           self.optimizer.step()

