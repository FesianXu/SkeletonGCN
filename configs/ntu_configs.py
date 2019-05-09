# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2018/10/17'
__version__ = ''

'''
Here is the local config for the relative euclidian distance in Two Tail Network
'''

import numpy as np

class GlobalConfig(object):
  __instance = None  # to implement singleton
  __call_cls_count = 0  # the counter of calling for new instances
  __is_init_finished = False
  __is_used_server = True  # use server
  def __init__(self):
    GlobalConfig.__call_cls_count += 1
    GlobalConfig.__is_init_finished = False
    '''
    Beginning the GLOBAL CONFIG
    '''

    self.num_action = 60
    self.max_person = 2  # the max person in a frame
    self.base_num_channel = 3
    self.num_channel = self.base_num_channel*self.max_person # here only the distance work
    self.max_nframe = 300
    self.num_clips = self.max_nframe
    self.is_single = False # the formation of the datapool
    self.is_cuda = True # whether use gpu or not
    self.cuda_id = 1
    self.batch_size = 8
    
    self.num_joints = 25
    self.resize_width = self.num_clips * 1
    self.resize_height = self.num_joints * 2

    self.connect_bone = [
        (3, 2), (2, 20), # head
        (20,4), (4,5), (5,6), (6,7), (7,21), (6,22), # left arm
        (20,8), (8,9), (9,10), (10,11), (11,23), (10,24), # right arm
        (20,1), (1,0), # spine
        (0,12), (12,13), (13,14), (14,15), # left foot
        (0,16), (16,17), (17,18), (18,19) # right foot
    ] # the joint id begin from 0
    
    self.total_actors = range(1, 41)  # actor id begin with 1, the total actor is 40 persons
    self.train_actors_config = np.array([1, 2, 4, 5, 8, 9, 13, 14, 15,
                                         16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38])
    self.test_actors_config = [i for i in self.total_actors if i not in self.train_actors_config]

    self.total_view = range(0, 5)
    self.train_view_config = np.array([0,2,4])
    self.test_view_config = [i for i in self.total_view if i not in self.train_view_config]
    
    self.ntu_root_path = '/home/fesian/AI_workspace/datasets/NTU/'
    if self.is_single:
      self.ntu_xsub_loading_list = '{}loading_list/x_sub_single.npy'.format(self.ntu_root_path)
      self.ntu_xview_loading_list = '{}loading_list/x_view_single.npy'.format(self.ntu_root_path)
    else:
      self.ntu_xsub_loading_list = '{}loading_list/x_sub.npy'.format(self.ntu_root_path)
      self.ntu_xview_loading_list = '{}loading_list/x_view.npy'.format(self.ntu_root_path)
    self.ntu_missing_list = '{}loading_list/missing_list.npy'.format(self.ntu_root_path)
    self.ntu_whole_raw_path = '{}whole_raw.npy'.format(self.ntu_root_path)
    self.ntu_data_path = '{}raw_npy/'.format(self.ntu_root_path)

    # about the mask transfer kernel size range
    self.kernel_range = [64, 64, 64, 32, 32, 32, 32, 32, 32]
    self.stgcn_params = [
        [64, 64],
        [64, 64],
        [64, 64],

        [64, 128],
        [128, 128],
        [128, 128],

        [128, 256],
        [256, 256],
        [256, 256],
    ]
    
    '''
    Ending the GLOBAL CONFIG
    '''
    GlobalConfig.__is_init_finished = True

  def get_model_path(self, model_root_path, comment, exp_num):
    return model_root_path % (comment, exp_num)

  def get_log_path(self, root_path, id):
    return root_path % (id)

  def get_exp_data_path(self, root_path, comment):
    return root_path % (comment)

  '''
  ABOVE ARE ALL CONFIG
  '''

  def __setattr__(self, key, value):
    if GlobalConfig.__is_init_finished is True:
      raise AttributeError('{}.{} is READ ONLY'.format(type(self).__name__, key))
    else:
      self.__dict__[key] = value

  def __new__(cls, *args, **kwargs):
    if GlobalConfig.__instance is None:
      GlobalConfig.__instance = object.__new__(cls)
    return GlobalConfig.__instance

  def callCounter(self):
    print('The Global Config class has been called for %d times' % self.__call_cls_count)
