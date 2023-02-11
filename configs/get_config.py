from yacs.config import CfgNode as CN
from datetime import datetime


_C = CN()
# basic configs
_C.name = 'default'
_C.author = 'Xu Wayen'
_C.output_path = 'default'
_C.seed = 1958
_C.training_steps = 20

# navigation config
_C.habitat_config_path = ''
_C.max_moves = 1000
_C.success_distance = 1.0
_C.use_office_reward = True

# backbone
_C.preprocess_model = 'resnet34'

# merge from files
_C.defrost()
_C.merge_from_file('configs/default.yaml')

if _C.output_path == 'DATE_TIME':
    _C.output_path = datetime.now().strftime("%Y%m%d_%H-%M%S")
_C.freeze() 


if __name__ == '__main__':
    print(_C)