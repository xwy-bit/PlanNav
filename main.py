from configs.get_config import _C as config
from utils.self_play import SelfPlay , SharedStorage , ReplayBuffer
import habitat
from algorithm.normal_mbase.policy import policy

# initialize worker & storage
self_play_worker = SelfPlay(config)
shared_storage = SharedStorage(config)
replay_buffer = ReplayBuffer(config)

trainable_agent = policy(config)
trainable_agent.train(replay_buffer,shared_storage)

breakpoint()
# TODO: check each config