# add parent dir to import path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from argparse import Namespace
import habitat
import numpy as np
from scipy.spatial.transform import Rotation as R

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
        # don't stop
        done = False
        return observations ,reward , done, info 
    
    def close(self):
        self.env.close()

    def get_parameters(self):
        position = self.sim.get_agent_state().position
        rotation_quaternion = self.sim.get_agent_state().rotation
        rotation_quaternion = np.array([rotation_quaternion.w
                                ,rotation_quaternion.x
                                ,rotation_quaternion.y
                                ,rotation_quaternion.z])
        object_rotation_matrix = R.from_quat(rotation_quaternion).as_matrix()
        y_rotation_angle  = -np.arctan2(object_rotation_matrix[0, 2], object_rotation_matrix[2, 2])
        current_pose = np.array([position[0],position[1],y_rotation_angle])
        depth_class = self.sim.sensor_suite.sensors['depth'].config
        rgb_class = self.sim.sensor_suite.sensors['rgb'].config
        param = {'dsensor_height':depth_class.position[1]
            ,'min_depth':depth_class.min_depth
            ,'max_depth':depth_class.max_depth
            ,'depth_hfov': depth_class.hfov
            ,'dframe_height':depth_class.height
            ,'dframe_width':depth_class.width
            ,'rframe_height':rgb_class.height
            ,'rframe_width':rgb_class.width
            ,'rsensor_height':rgb_class.position[1]
            ,'rgb_hfov':rgb_class.hfov
            ,'agent_rotation_quaternion':rotation_quaternion
            ,'agent_rotation_y':y_rotation_angle
            ,'current_pose':current_pose}
        param = Namespace(**param)
        return param
if __name__ == '__main__':
    from configs.get_config import _C as config
    Env = HabitatGame(config)
    observations = Env.reset()
    for ii in range(3):
        observations = Env.step(3)
        param = Env.get_parameters()
        print(param.agent_position)
    print('finish test')