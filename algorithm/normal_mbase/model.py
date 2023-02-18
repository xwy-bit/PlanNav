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
        self.layer_num = config.normal_mbase.policynet_layer_num
        self.f1 = NN.Linear(self.layer_num[0],self.layer_num[1])
        self.f2 = NN.Linear(self.layer_num[1],self.layer_num[2])
        self.f3 = NN.Linear(self.layer_num[2],self.layer_num[3])
        self.f4 = NN.Linear(self.layer_num[3],self.layer_num[4])
        self.Fun = F.gelu
        self.softmax = NN.Softmax(0)
        self.optimizer = torch.optim.Adam(
            self.parameters(),lr =config.normal_mbase.policy_train_lr)
    def forward(self,state):
        hidden = self.f1(state)
        hidden = self.Fun(hidden)
        hidden = self.f2(hidden)
        hidden = self.Fun(hidden)
        hidden = self.f3(hidden)
        hidden = self.Fun(hidden)
        action_prob_ = self.f4(hidden)

        return self.softmax(action_prob_)

class ValueNet(NN.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.layer_num = config.normal_mbase.valuenet_layer_num
        self.f1 = NN.Linear(self.layer_num[0],self.layer_num[1])
        self.f2 = NN.Linear(self.layer_num[1],self.layer_num[2])
        self.f3 = NN.Linear(self.layer_num[2],self.layer_num[3])
        self.f4 = NN.Linear(self.layer_num[3],self.layer_num[4])
        self.Fun = F.gelu        
   
    def forward(self,input):
        hidden = self.f1(input)
        hidden = self.Fun(hidden)
        hidden = self.f2(hidden)
        hidden = self.Fun(hidden)
        hidden = self.f3(hidden)
        hidden = self.Fun(hidden)
        value = self.f4(hidden)
        return value

class ActorCritic:
    def __init__(self,config) -> None:
        self.actor = PolicyNet(config)
        self.critic = ValueNet(config)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.gamma = config.discount_rate

    def take_action(self,state):
        '''
        state: [torch.Tensor] feature of state
        '''
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    def update(self,trainsition_pair):
        states , actions , rewards ,next_states = trainsition_pair
        td_target = rewards + self.gamma * self.critic(next_states)
        td_delta = td_target - self.critic(states)
        log_probs = torch.log(self.actor(states).gather(1,actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()