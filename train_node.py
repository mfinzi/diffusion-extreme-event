from absl import logging
import jax
from jax import jit
from jax import random
from jax import vmap
import jax.numpy as jnp
import ml_collections
import numpy as np
from functools import partial  # pylint: disable=g-importing-member
import os
import pickle
import time
from flax import linen as nn
from jax.experimental.ode import odeint
from oil.utils.utils import FixedNumpySeed
from oil.logging.lazyLogger import LazyLogger
import ode_datasets
from typing import Sequence


import optax
from tqdm.auto import tqdm
import tensorflow as tf

import warnings
import copy
from oil.tuning.study import Study
from oil.tuning.args import argupdated_config
import os
import pandas as pd

class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.swish(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x

# class NeuralODE(nn.Module):
#     T: Sequence[float]
#     @nn.compact
#     def __call__(self, x):
#         F = MLP([128,128,x.shape[-1]])
#         xt = odeint(lambda z,t: F(z), x, self.T, rtol=1e-3).transpose((1,0,2))
#         return x


def train(*,seed=37,epochs=100,dataset_timesteps=60,ds=4000,bs=1000,lr=1e-3,chunk_size=10,
    ic_conditioning=False,dataset='LorenzDataset',log_dir=None,log_suffix=''):
    # construct dataloaders and dataset
    timesteps = dataset_timesteps
    with FixedNumpySeed(seed):
        ds = getattr(ode_datasets, dataset)(N=ds + bs)
    if dataset == 'NPendulum':
        trajectories = ds._Zs[bs:, :timesteps]
    else:
        trajectories = ds.Zs[bs:, :timesteps]
    # chunk the trajectories
    trajectories = trajectories.reshape(-1,chunk_size,*trajectories.shape[2:])
    T_long = ds.T_long[:timesteps]  # pylint: disable=invalid-name
    T_short = T_long[:chunk_size]-T_long[0]
    dataset = tf.data.Dataset.from_tensor_slices(trajectories)
    dataiter = dataset.shuffle(len(dataset)).batch(bs).as_numpy_iterator
    
    

    key = random.PRNGKey(42) if seed is None else random.PRNGKey(seed)
    key, init_seed = random.split(key)
    #model = NeuralODE(T=T_short)
    model = MLP([128,128,trajectories.shape[-1]])
    params = model.init(init_seed, x=trajectories[0])

    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(params)

    @jit
    def loss(params, x):
        x0 = x[:,0]
        F = lambda x,t: model.apply(params,x)/(T_short[-1])
        x_pred = odeint(F, x0, T_short, rtol=1e-3).transpose((1,0,2))
        return jnp.mean(jnp.abs(x-x_pred))

    loss_grad_fn = jax.value_and_grad(loss)

    @jit
    def update_fn(params,  opt_state, key, data):
        loss_val, grads = loss_grad_fn(params, data)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        key, _ = random.split(key)
        return params, opt_state, key, loss_val
    # training loop
    for epoch in tqdm(range(epochs + 1)):
        for data in dataiter():
            params, opt_state, key, loss_val = update_fn(
                params, opt_state, key, data)
        if epoch % 5 == 0:
            message = f'Loss epoch {epoch}: {loss_val:.5f}'
            logging.info(message)
            print(message)
    metrics = pd.DataFrame({'loss':np.array(loss_val)},index=[0])
    return params,metrics
    

def NodeTrial(cfg,i=None):
    tf.config.experimental.set_visible_devices([], "GPU")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if i is not None:
            orig_suffix = cfg.get('log_suffix','')
            if orig_suffix is None: orig_suffix = ''
            cfg['log_suffix'] = os.path.join(orig_suffix,f'trial{i}/')
        params, metrics = train(**cfg)
        logger = LazyLogger(cfg.get("log_dir",None),cfg.get("log_suffix",""))
        logger.save_object(cfg,'config')
        logger.save_object(params,'params')
        return cfg,metrics

# if __name__ == "__main__":
#   # Provide access to --jax_backend_target and --jax_xla_backend flags.
#   tf.config.experimental.set_visible_devices([], "GPU")
#   with warnings.catch_warnings():
#     cfg = copy.deepcopy(train.__kwdefaults__)
#     cfg.update({'epochs':100,'dataset':'FitzHughDataset','lr':1e-3})
#     print(NodeTrial(cfg)[1])

if __name__=='__main__':
  with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=UserWarning)
      cfg_spec = copy.deepcopy(train.__kwdefaults__)
      cfg_spec.update({
          'study_name':'node_all','dataset':['LorenzDataset','FitzHughDataset','NPendulum'],
          'epochs':500,
      })
      
      cfg_spec = argupdated_config(cfg_spec)
      name = cfg_spec.pop('study_name')#['study_name']#.pop('study_name')
      basedir = cfg_spec.get('log_dir','.')
      if basedir is None: basedir = '.'
      cfg_spec['log_dir'] = os.path.join(basedir, name)
      thestudy = Study(NodeTrial,cfg_spec,study_name=name,
              base_log_dir=basedir)
      thestudy.run(ordered=True)
      print(thestudy.covariates())
      print(thestudy.outcomes)