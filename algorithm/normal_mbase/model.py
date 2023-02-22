import torch
import torch.nn as NN
import torch.nn.functional as F

class EnvNet(NN.Module):
    def __init__(self,config,writer) -> None:
        super().__init__()
        self.layer_num = config.normal_mbase.envnet_layer_num
        self.f1 = NN.Linear(self.layer_num[0],self.layer_num[1])
        self.f2 = NN.Linear(self.layer_num[1],self.layer_num[2])
        self.f3 = NN.Linear(self.layer_num[2],self.layer_num[3])
        self.f4 = NN.Linear(self.layer_num[3],self.layer_num[4],bias=False)
        self.Fun = F.gelu
        self.lossfun = NN.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.parameters(),lr = config.normal_mbase.envnet_train_lr)
        self.writer = writer
        self.writer_counter = 0
    def forward(self,action,last_state):
        '''
        concat action & last_state
            action    : B * N
            last_state: B * M 
        '''

        hidden = torch.concat([action,last_state],1)
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

        # record loss
        self.writer.add_scalar('env loss',loss,self.writer_counter)
        self.writer_counter += 1

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
    def __init__(self,config,writer) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actor = PolicyNet(config).to(self.device)
        self.critic = ValueNet(config).to(self.device)
        
        self.writer = writer
        self.writer_counter = 0

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.gamma = config.discount_rate

    def take_action(self,state):
        '''
        state: [torch.Tensor] feature of state
        '''
        probs = self.actor(state.detach_())
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item() + 1 # +1 for not take stop
    def update(self,trainsition_pair):
        states , actions , rewards ,next_states = trainsition_pair
        td_target = rewards + self.gamma * self.critic(next_states.detach())
        td_delta = td_target - self.critic(states.detach())
        log_probs = torch.log(self.actor(states.detach()).gather(1,actions.detach()))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        # record loss
        self.writer.add_scalar('actor_loss',actor_loss,self.writer_counter)
        self.writer.add_scalar('critic_loss',critic_loss,self.writer_counter)
        self.writer_counter += 1

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()