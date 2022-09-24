import logging
import functools
import os
import pathlib
import re
import sys
import warnings
import matplotlib, PIL
import argparse
import pdb

try:
    import rich.traceback
    rich.traceback.install()
except ImportError:
    pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml
from tensorflow import keras
import tensorflow as tf
from dreamerv2.common.replay import convert

import agent
import common


def main():

    def make_env(mode):
        suite, task = config.task.split('_', 1)
        if suite == 'dmc':
            env = common.DMC(task, config.action_repeat, config.render_size, config.dmc_camera)
            env = common.NormalizeAction(env)
        elif suite == 'atari':
            env = common.Atari(task, config.action_repeat, config.render_size,
                               config.atari_grayscale)
            env = common.OneHotAction(env)
        elif suite == 'crafter':
            assert config.action_repeat == 1
            outdir = logdir / 'crafter' if mode == 'train' else None
            reward = bool(['noreward', 'reward'].index(task)) or mode == 'eval'
            env = common.Crafter(outdir, reward)
            env = common.OneHotAction(env)
        elif suite == 'taxi':

            from visgrid.envs import TaxiEnv
            from visgrid.wrappers.transforms import NoiseWrapper, ClipWrapper, TransformWrapper
            env = TaxiEnv(size=5,
                          n_passengers=1,
                          exploring_starts=True,
                          terminate_on_goal=True,
                          fixed_goal=False,
                          depot_dropoff_only=True,
                          should_render=True,
                          dimensions=TaxiEnv.dimensions_5x5_to_64x64)
            env = NoiseWrapper(env, sigma=0.01)
            env = ClipWrapper(env, 0.0, 1.0)
            env = TransformWrapper(env, lambda x: x - 0.5)
            env = common.GymWrapper(env)
            env = common.ResizeImage(env)

            if hasattr(env.act_space['action'], 'n'):
                env = common.OneHotAction(env)
            else:
                env = common.NormalizeAction(env)
        else:
            raise NotImplementedError(suite)
        env = common.TimeLimit(env, config.time_limit)
        return env

    pdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str)


    args = parser.parse_args()

    logdir = pathlib.Path(args.model_path).expanduser()

    #check if logdir is valid
    if not logdir.exists() or not (logdir/'saved_models').exists() or os.listdir((logdir/'saved_models'))==0:
        raise NotImplementedError

    #create reconstruction save dir
    (logdir/'reconstructions').mkdir(parents=True, exist_ok=True)

    #store the configs path
    config_path = logdir/'config.yaml'
    configs = yaml.safe_load(config_path.read_text())
    #parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)

    config = common.Config(configs)
    #for name in parsed.configs:
    #    config = config.update(configs[name])

    #config = common.Flags(config).parse(remaining)


    tf.config.run_functions_eagerly(config.jit)
    message = 'No GPU found. To actually train on CPU remove this assert.'
    assert tf.config.experimental.list_physical_devices('GPU'), message
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        from keras import mixed_precision as prec
        prec.set_global_policy(prec.Policy('mixed_float16'))
    
    #setup train replay and step counter
    train_replay = common.Replay(logdir / 'train_episodes', **config.replay)
    step = common.Counter(train_replay.stats['total_steps'])

    num_eval_envs = min(config.envs, config.eval_eps)
    if config.envs_parallel == 'none':
        train_envs = [make_env('train') for _ in range(config.envs)]
        eval_envs = [make_env('eval') for _ in range(num_eval_envs)]
    else:
        make_async_env = lambda mode: common.Async(functools.partial(make_env, mode), config.
                                                   envs_parallel)
        train_envs = [make_async_env('train') for _ in range(config.envs)]
        eval_envs = [make_async_env('eval') for _ in range(eval_envs)]


    #creating gym environment and random agent
    act_space = train_envs[0].act_space
    obs_space = train_envs[0].obs_space

    random_agent = common.RandomAgent(act_space)
    driver = MiniDriver(train_envs)
    driver.reset()


    for saved_model in (logdir/'saved_models').iterdir():

        agnt = agent.Agent(config, obs_space, act_space, step)
        agnt.load(saved_model)
	pdb.set_trace()
        traject_list = driver(random_agent, train_step)

        for trajectory in traject_list:

            trajectory = agnt.wm.preprocess(trajectory)

            for key in agent.wm.heads['decoder'].cnn_keys:
                name = key.replace('/', '_')
                reconstruction_prediction(agnt, trajectory, key)



    
    def train_step(tran):

        trajectory = {
                k: np.expand_dims(np.array([convert(experience[k]) for experience in [tran]]),0) for k in tran.keys()}

        return trajectory

        


    def reconstruction_prediction(agnt, trajectory, key):
        
        embed = agnt.wm.encoder(trajectory)
        states, _ = agnt.wm.rssm.observe(embed, trajectory['action'], trajectory['is_first'])

        recon = agnt.wm.heads['decoder'](agnt.wm.rssm.get_feat(states))[key].mode()
        init = {k: v[:, -1] for k, v in states.items()}

        prior = agnt.wm.rssm.imagine(trajectory['action'], init)
        openl = agnt.wm.heads['decoder'](agnt.wm.rssm.get_feat(prior))[key].mode()

        model_prediction = tf.concat([recon + 0.5, openl + 0.5], 1)
        ground_truth = trajectory[key] + 0.5

        error = (model_prediction - ground_truth + 1)/2
        image = tf.concat([ground_truth, model_prediction, error], 2)
        B, T, H, W, C = image.shape
        image.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))


    
    



class MiniDriver:

  def __init__(self, envs):
    self._envs = envs
    self._act_spaces = [env.act_space for env in envs]
    self.reset()

  def reset(self):
    self._obs = [None] * len(self._envs)
    self._state = None

  def __call__(self, policy, train_step):
    
    obs = {i: self._envs[i].reset() for i, ob in enumerate(self._obs) if ob is None or ob['is_last']}
      
    
    for i, ob in obs.items():
        self._obs[i] = ob() if callable(ob) else ob
        act = {k: np.zeros(v.shape) for k, v in self._act_spaces[i].items()}
        tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
        

    obs = {k: np.stack([o[k] for o in self._obs]) for k in self._obs[0]}
     
    actions, self._state = policy(obs, self._state)
    
    actions = [
          {k: np.array(actions[k][i]) for k in actions}
          for i in range(len(self._envs))]
    assert len(actions) == len(self._envs)
    obs = [e.step(a) for e, a in zip(self._envs, actions)]
      
    obs = [ob() if callable(ob) else ob for ob in obs]
    traj_list = []
    for i, (act, ob) in enumerate(zip(actions, obs)):
        
        tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
        traj = train_step(tran)
        traj_list.append(traj)
        
        
    self._obs = obs
    return traj_list
   

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
      return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
      return value.astype(np.uint8)
    return value


if __name__ == '__main__':
    main()
