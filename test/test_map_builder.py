import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np 
from algorithm.rl_map.map_builder import MapBuilder
from configs.get_config import _C as config
from common.games import HabitatGame
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
from habitat.utils.visualizations import maps

Env = HabitatGame(config)

map_builder = MapBuilder(config)

observations , reward , done , info = Env.reset()
observations , reward , done , info = Env.reset()
point_cloud_all = np.zeros([1,3],dtype=np.float64)
for ii in range(36):
    observations , reward , done , info = Env.step(3)
    grid_map = map_builder.update_map(observations['depth'],Env)

    grid_map = np.array(grid_map > 100,dtype=np.int32)
    fig = plt.figure()
    plt.imsave('test/asset/grid_map.png',grid_map)
    plt.close()
    plt.imsave('test/asset/rgb_image.png' , observations['rgb'],cmap='gray')
    plt.close()
    GT_map = maps.colorize_topdown_map(info['top_down_map']['map'])
    # plt.imshow(GT_map,interpolation="nearest", origin="upper")
    plt.imsave('test/asset/gt_map.png',GT_map,cmap='gray')
    plt.close()
    depth = observations['depth']
    depth_norm_img = img_as_ubyte(depth).squeeze()
    plt.imsave('test/asset/depth_image.png' ,depth_norm_img ,cmap='gray')
    plt.close()
