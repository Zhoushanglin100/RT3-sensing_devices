'''
    This file is for the input of RL controller.
    This program can be used for BP+PP, BP+rPP, rBP+PP
    When ours = True, the program is for BP+PP;
    if ours=False and random_pattern=True, the program is for BP+rPP;
    if ours=False and random_pattern=False, the program is for rBP+PP.
    The models is in the Baidu Netdisk.
'''
# todo: the name of models in the baidu netdisk should be modified

import torch
from mnist_generate_rl_input import random_generate_rl_input, extract_generate_rl_input
from Pruning.mnist_precompression_extract_joint_training import model
from utils.load_config_file import load_config_file

block_size = 10
pruning_number_list = [10,30,50,70,80,90]
# pruning_number_list = [100,1300,2500,3700,5000,6200,7500,8700,9300,9900]

ours = True
random_pattern = False
if ours:#True
    print('#' * 89)
    print('A.pattern pruning from precompression model')
    print('B.extract important pattern from precompression model')
    print('C.training(pruning number={})'.format(pruning_number_list))
    print('#' * 89)

    config_file = './config_file/mnist_prune_ratio.yaml'
    prune_ratios = load_config_file(config_file)
    model.load_state_dict(torch.load("../LeNet5-MNIST-PyTorch/models/mnist_0.99.pt"))
    para_set = extract_generate_rl_input(model,block_size,prune_ratios,pruning_number_list)

# else:
#     if random_pattern:#False True
#         print('#' * 89)
#         print('A.pattern pruning from precompression model')
#         print('B.random generate pattern for every layer')
#         print('C.training(pruning number={})'.format(pruning_number_list))
#         print('#' * 89)

#         config_file = './config_file/prune_ratio_v6.yaml'
#         prune_ratios = load_config_file(config_file)
#         model.load_state_dict(torch.load('./model/model_after_BP.pt'))
#         para_set = random_generate_rl_input(prune_ratios,pruning_number_list,block_size)
#     else:#False False
#         print('#' * 89)
#         print('A.pattern pruning from random column pruning model(10epochs)')
#         print('B.extract pattern from random column pruning model')
#         print('C.training(pruning number={})'.format(pruning_number_list))
#         print('#' * 89)

#         config_file = './config_file/prune_ratio_v1.yaml'
#         prune_ratios = load_config_file(config_file)
#         model.load_state_dict(torch.load('./model/model_after_rBP.pt'))
#         para_set = extract_generate_rl_input(model,block_size,prune_ratios,pruning_number_list)


controller_params = {
    "model": model,
    "sw_space":(para_set),
    "level_space":([[[0, 1, 2], [0, 1, 3], [0, 1, 4], 
                     [0, 2, 3], [0, 2, 4], 
                     [0, 3, 4], 
                     [1, 2, 3], [1, 2, 4], 
                     [1, 3, 4], 
                     [2, 3, 4],
                     ]]),
    "num_children_per_episode": 1,
    'hidden_units': 35,
    'max_episodes': 300,
    'epochs':1,
    "timing_constraint":20 #115 for high, 104 for middle,94 for low
}