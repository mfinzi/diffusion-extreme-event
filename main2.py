# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main file for running the example.

This file is intentionally kept short. The majority for logic is in libraries
than can be easily tested and imported in Colab.
"""

# Required import to setup work units when running through XManager.
#from clu import platform
import jax
from ml_collections import config_flags
import tensorflow as tf
import train
from config import get_config,config
import warnings
import copy
from oil.tuning.study import Study
from oil.tuning.args import argupdated_config
import os
import pandas as pd

def Trial(cfg,i=None):
    tf.config.experimental.set_visible_devices([], "GPU")
    with warnings.catch_warnings():
        
        warnings.simplefilter("ignore")
        if i is not None:
            orig_suffix = cfg.get('log_suffix','')
            cfg['log_suffix'] = os.path.join(orig_suffix,f'trial{i}/')
        outcome = train.train_and_evaluate(config(**cfg), '.')
        return cfg,outcome

# if __name__ == "__main__":
#   # Provide access to --jax_backend_target and --jax_xla_backend flags.
#   tf.config.experimental.set_visible_devices([], "GPU")
#   with warnings.catch_warnings():
#     cfg = get_config()
#     cfg.update({'epochs':100,'dataset':"NPendulum",'channels':24,'ic_conditioning':True})
#     print(Trial(cfg)[1])



if __name__=='__main__':
  with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=UserWarning)
      cfg_spec = copy.deepcopy(dict(**get_config()))
      cfg_spec.update({
          'study_name':'lorenz_long','dataset':['LorenzDataset','FitzHughDataset','NPendulum'],
          'ic_conditioning':[False,True], 'epochs':100000, 'channels':24,# 'ds':10000,
      })
      
      cfg_spec = argupdated_config(cfg_spec)
      name = cfg_spec['study_name']#.pop('study_name')
      basedir = cfg_spec.get('log_dir','.')
      cfg_spec['log_dir'] = os.path.join(cfg_spec.get("log_dir","") ,name)
      thestudy = Study(Trial,cfg_spec,study_name=name,
              base_log_dir=basedir)
      thestudy.run(ordered=True)
      print(thestudy.covariates())
      print(thestudy.outcomes)