# add parent dir to import path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from algorithm.rl_map.utils.depth_utils import get_camera_matrix,get_point_cloud_from_z
from common.games import HabitatGame
from configs.get_config import _C as config
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.util import img_as_ubyte
import numpy as np

Env = HabitatGame(config)

observations = Env.reset()
observations = Env.reset()
for ii in range(30):
    observations , reward , done , info = Env.step(3)
    depth  = np.array(observations['depth']).squeeze()

    camera_intrinsics = get_camera_matrix(256,256,90)
    print(np.shape(depth))
    point_cloud = get_point_cloud_from_z(depth,camera_intrinsics)
    print(np.mean(point_cloud[...,0] + point_cloud[...,1]))
    np.save('test/asset/point_cloud.npy',point_cloud)
    with open('test/asset/point_cloud.txt','w+') as f:
        print(point_cloud,file=f)
    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[...,0], point_cloud[...,1], point_cloud[...,2],s = 1, c=point_cloud[...,2])
    # ax.set_xlim3d(-1,1)
    # ax.set_ylim3d(0,1)
    # ax.set_zlim3d(-1,1)
    ax.view_init(40, -45)
    plt.savefig('test/asset/point_cloud.png')
    plt.imsave('test/asset/rgb_image.png' , observations['rgb'],cmap='gray')
    depth_norm_img = img_as_ubyte(depth).squeeze()
    plt.imsave('test/asset/depth_image.png' ,depth_norm_img ,cmap='gray')

print('finish test')
