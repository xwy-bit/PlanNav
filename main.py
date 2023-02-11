from configs.get_config import _C as config
from utils.self_play import SelfPlay , SharedStorage , ReplayBuffer
import habitat

# initialize worker & storage
self_play_worker = SelfPlay(config)
shared_storage = SharedStorage(config)
replay_buffer = ReplayBuffer(config)

self_play_worker.continuous_self_play(shared_storage,replay_buffer)
