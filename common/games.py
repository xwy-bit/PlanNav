# add parent dir to import path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import habitat
from configs.get_config import _C as config

class HabitatGame:
    def __init__(self,common_config) -> None:
        
        self.habitat_config = habitat.get_config(common_config.habitat_config_path)
        self.config = common_config
        self.env = habitat.Env(self.habitat_config)
        observations = self.env.reset() # first observation 
        self.sim = self.env._sim
        
    def reset(self):
        observations = self.env.reset()
        done = False
        reward = 0.0
        info  = {}
        info['scene_name'] = self.env.current_episode.scene_id
        info['num_steps'] = 0
        return observations , reward , done , info
    
    def step(self,action):
        '''
        observations: 'rgb' 'depth' 'pointgoal_with_gps_compass'
        metrics: 'distance_to_goal' 'success' 'spl' 'distance_to_goal_reward'
        '''
        observations = self.env.step(action)
        
        metrics = self.env.get_metrics()
        
        # done if distance to goal less than 1.0
        if metrics['distance_to_goal'] < config.success_distance:
            done = True
        else:
            done = False
        
        done = done or self.env.episode_over
        collision = self.sim.previous_step_collided        
        
        # use office reward or not    
        if self.config.use_office_reward:
            reward = metrics['distance_to_goal_reward']
        elif done:
            reward = 1.0
        else:
            reward = 0.0
            
        info = metrics
        info['collision'] = collision
        if collision:
            reward = reward - 1
        if action != 1:
            reward = reward - 0.5
        
        # debug
        print(info)
        return observations ,reward , done, info 
    
    def close(self):
        self.env.close()

if __name__ == '__main__':
    Env = HabitatGame('configs/habitat_default.yaml')
    observations = Env.reset()
    for ii in range(3):
        observations = Env.step(2)
        print(list(observations[0].keys()))
    print('finish test')