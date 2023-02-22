from yacs.config import CfgNode as CN
from datetime import datetime


_C = CN()
# basic configs
_C.name = 'default'
_C.author = 'Xu Wayen'
_C.output_path = 'default'
_C.seed = 1958
_C.play_steps = 20 # play steps within a call

# navigation config
_C.habitat_config_path = ''
_C.max_moves = 1000
_C.success_distance = 1.0
_C.use_office_reward = True
_C.discount_rate = 0.98

# backbone
_C.preprocess_model = 'resnet34'

# model-based AC
_C.normal_mbase = CN()

_D = _C.normal_mbase
_D.envnet_layer_num = []
_D.policynet_layer_num = []
_D.valuenet_layer_num = []
_D.envnet_train_epochs = 1
_D.policynet_train_epochs = 1
_D.envnet_train_sample_num = 1
_D.envnet_train_batchsize = 1
_D.policynet_train_sample_num = 1
_D.policy_train_batchsize = 1
_D.envnet_train_lr = 0.1
_D.envnet_train_momentun = 0.9
_D.policy_train_lr = 0.9
_D.envpolicy_train_epoches = 1
_D.policy_train_epoches = 1
_D.env_train_epoches = 1
_D.env_train_batchsize = 1


# merge from files
_C.defrost()
_C.merge_from_file('configs/default.yaml')

if _C.output_path == 'DATE_TIME':
    _C.output_path = datetime.now().strftime("%Y%m%d_%H-%M%S")
_C.freeze() 


if __name__ == '__main__':
    print(_C)