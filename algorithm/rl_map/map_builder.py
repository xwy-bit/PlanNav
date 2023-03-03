import numpy as np
import algorithm.rl_map.utils.depth_utils as du
import torch.nn as nn

class MapBuilder(object):
    def __init__(self,config) -> None:
        # TODO: get config from environment
        self.config = config
        self.shift_loc = \
            [self.config.vision_range * self.config.map_resolution //2,0,np.pi /2.0]
        self.max_height = config.max_height
        self.min_height = config.min_height
        self.resolution = config.map_resolution
        self.vision_range = config.vision_range
        self.du_scale = config.du_scale
        self.grid_num = config.grid_num
        self.grid_smooth = np.zeros([config.grid_num,config.grid_num],dtype=np.int32)

    def update_map(self,depth,env):
        
        param = env.get_parameters()
        depth  = np.array(depth).squeeze()
        depth = depth  * (param.max_depth - param.min_depth) + param.min_depth
        camera_matrix = du.get_camera_matrix(param.dframe_width
                                            ,param.dframe_height,param.depth_hfov)
        point_cloud = du.get_point_cloud_from_z(depth,camera_matrix).squeeze()
        
        agent_view = du.transform_camera_view(point_cloud,
                                            param.dsensor_height,0)
                                   
        agent_view_centered = du.transform_pose(agent_view,param.current_pose)
        
        
        return self.pcloud_to_grid(agent_view_centered.reshape([-1,3]))
    def pcloud_to_grid(self,point_cloud):
        '''
        point_cloud: [SHAPE] -1,3
        '''
        point_num = point_cloud.shape[0]
        for index in range(point_num):
            coordinate = point_cloud[index,:]
            if coordinate[2] < self.max_height and coordinate[2] > self.min_height:
                self.grid_smooth[int(coordinate[0]/self.resolution) + self.grid_num //2
                    ,int(coordinate[1]/self.resolution) + self.grid_num //2 ] += 1
        return self.grid_smooth
            