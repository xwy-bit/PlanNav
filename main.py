from configs.get_config import _C as config
from utils.self_play import SelfPlay , SharedStorage , ReplayBuffer
import habitat
from algorithm.normal_mbase.policy import policy
from torch.utils.tensorboard import SummaryWriter
import os 
# tensorboard logs
log_path = 'logs/'+ config.output_path
os.mkdir(log_path)
writer = SummaryWriter(log_path)

# initialize worker & storage
shared_storage = SharedStorage(config)
replay_buffer = ReplayBuffer(config)

trainable_agent = policy(config,writer)
trainable_agent.train(replay_buffer,shared_storage)

breakpoint()
# TODO: check each config