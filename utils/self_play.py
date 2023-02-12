import torch
import copy
import numpy as np
from common.games import HabitatGame
import habitat

class SharedStorage:
    '''
        get and set checkpoints
    '''
    def __init__(self,config,checkpoint = {}) -> None:
        self.config = config
        self.current_checkpoint = copy.deepcopy(checkpoint)
    def save_checkpoint(self,path = None):
        if not path:
            path = self.config.output_path + '/model.checkpoint'
            
        torch.save(self.current_checkpoint, path)
        
    def get_checkpoint(self):
        return copy.deepcopy(self.current_checkpoint)
    
    def get_info(self,keys):
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys,list):
            return {key: self.current_checkpoint[key] for key in keys}
        else :
            raise TypeError
    def set_info(self,keys,values = None):
        if isinstance(keys,str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys,dict):
            self.current_checkpoint.uptdate(keys)
        else:
            raise TypeError

class GameHistory:
    def __init__(self) -> None:
        self.observations_history = {'rgb':[] , 'depth':[]} # for multimodality observations
        self.actions_history = []
        self.rewards_history = []
        self.infos_history = []
        self.scene_name = ''
        self.len = 0
    def get_observations(self,kind = 'rgb',index = -1, stacked_num = 1):
        '''
        kind : str -'rgb'/'depth' , list - ['rgb' ,'depth']
        index : last index of stacked observations
        '''
        current_len =  self.actions_history.shape[0]
        if index >= current_len:
            raise IndexError('index {} out of range {}'.format(index,current_len))
        
        # single type of key      
        if isinstance(kind,str):
            if kind == 'rgb':
                return self.observations_history['rgb'][
                    index%current_len + 1 - stacked_num:index%current_len + 1]
            elif kind == 'depth':
                                return self.observations_history['depth'][
                    index%current_len + 1 - stacked_num:index%current_len + 1]
            else:
                raise TypeError
        # multi-type of keys    
        elif isinstance(kind,list):
            if set(kind) == set(['rgb','depth']):
                stacked_rgbs = self.observations_history['rgb'][
                    index%current_len + 1 - stacked_num:index%current_len + 1]
                stacked_depths = self.observations_history['depth'][
                    index%current_len + 1 - stacked_num:index%current_len + 1]
                return stacked_rgbs , stacked_depths
            else:
                raise TypeError

    # @property
    # def len(self):
    #     length = np.shape(self.actions_history)[0]
        
    #     # navigation specialization
    #     if self.observations_history.__contains__('rgb'):
    #         assert length == np.shape(self.actions_history['rgb'])[0]
    #     if self.observations_history.__contains__('depth'):
    #         assert length == np.shape(self.actions_history['depth'])[0]
            
    #     return length
                   
      
class ReplayBuffer:
    def __init__(self,config , init_checkpoint = {} , initial_buffer = {}) -> None:
        self.config = config
        self.game_buffer = copy.deepcopy(initial_buffer) # as a dict
        if init_checkpoint:
            self.num_played_games = init_checkpoint['num_played_games']
            self.num_played_steps = init_checkpoint['num_played_games']
        else:
            self.num_played_games = 0
            self.num_played_steps = 0
                
        
        np.random.seed(self.config.seed)
        
    def save_game(self, game_history , shared_storage = None):
        self.game_buffer[self.num_played_games] = game_history
        self.num_played_games += 1
        self.num_played_steps += game_history.len
        
        # store number of games & steps 
        if shared_storage:
            shared_storage.set_info('num_played_games', self.num_played_games)
            shared_storage.set_info('num_played_steps', self.num_played_steps)
    
    def get_buffer(self):
        return self.game_buffer
    
    def sample_games(self,game_num):
        game_ids = []
        games = []
        for game_id, game_history in self.buffer.items():
            game_ids.append(game_id)
            games.append(game_history.game_priority) 
                   
        selected_game_ids = np.random.choice(
            list(self.game_buffer.keys())
            ,game_num)
        
        ret =[
            (game_id,self.game_buffer[game_id]) 
            for game_id in selected_game_ids
        ]
        return ret
        
        
class SelfPlay:
    def __init__(self,config) -> None:
        self.config = config
        self.Game = HabitatGame(config)

        # FIx random seed
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        # model 
        
    def continuous_self_play(self,shared_storage,replay_buffer,test_mode=False):
        shared_storage.set_info('training_step' , 0)
        while shared_storage.get_info('training_step')  <  self.config.training_steps :
            
            # TODO add model
            
            if not test_mode:
                game_history = self.play_game()

                replay_buffer.save_game(game_history,shared_storage)
                shared_storage.set_info('training_step',shared_storage.get_info('training_step') + 1)
        self.close_game()            
            
    def play_game(self):
        
        # initialize game history
        game_history = GameHistory()
        observation , reward , done , info = self.Game.reset()
        game_history.actions_history.append(4)
        game_history.observations_history['rgb'].append(observation['rgb'])
        game_history.observations_history['rgb'].append(observation['depth'])
        game_history.rewards_history.append(reward)
        game_history.infos_history.append(info)
        
        
        # TODO get scene name
        game_history.scene_name = info['scene_name']
        
        while not done and info['num_steps'] < self.config.max_moves:
            
            # TODO get actions
            action = 1
            
            observation , reward , done , info = self.Game.step(action)
            
            
            # add game history
            game_history.actions_history.append(action)
            game_history.observations_history['rgb'].append(observation['rgb'])
            game_history.observations_history['rgb'].append(observation['depth'])
            game_history.rewards_history.append(reward)
            game_history.len = info['num_steps']
            
        return game_history
    def close_game(self):
        self.Game.close()
            
        
     