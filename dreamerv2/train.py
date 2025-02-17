import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings
import pandas as pd
import argparse

from visgrid.wrappers.transforms import NoiseWrapper, ClipWrapper, TransformWrapper
from gym.wrappers.time_limit import TimeLimit

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

# import tensorflow as tf
# tf.config.run_functions_eagerly(True)
# tf.debugging.experimental.enable_dump_debug_info("/tmp/tfdbg2_logdir",
#                                                  tensor_debug_mode="FULL_HEALTH",
#                                                  circular_buffer_size=-1)

import agent
import common
import pdb

#pdb.set_trace()
if os.path.exists('./reward_data.csv'):
    reward_tracker = pd.read_csv('reward_data.csv')
else:
    reward_tracker = pd.DataFrame(columns = ['actual_reward', 'is_timeout', 'pred_reward_mode', 'pred_reward_mean', 'pred_discount_mode', 'pred_discount_mean','taxi_row', 'taxi_col', 'p_row', 'p_col','in_taxi'])

def main():
  #pdb.set_trace()
  configs = yaml.safe_load((
      pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)
  #pdb.set_trace()

  config = common.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_steps', type=int, required=True)
  args = parser.parse_args(remaining)

  config.update({'steps':args.num_steps})
  # config = common.Flags(config).parse(remaining)
  
  #pdb.set_trace()

  logdir = pathlib.Path(config.logdir).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  config.save(logdir / 'config.yaml')
  print(config, '\n')
  print('Logdir', logdir)

  import tensorflow as tf

  tf.config.run_functions_eagerly(config.jit)
  message = 'No GPU found. To actually train on CPU remove this assert.'
  assert tf.config.experimental.list_physical_devices('GPU'), message
  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    from keras import mixed_precision as prec
    prec.set_global_policy(prec.Policy('mixed_float16'))

  train_replay = common.Replay(logdir / 'train_episodes', **config.replay)
  #eval_replay = common.Replay(logdir / 'eval_episodes', **dict(
  #    capacity=config.replay.capacity // 10,
  #    minlen=config.dataset.length,
  #    maxlen=config.dataset.length))
  eval_replay = common.Replay(logdir/'eval_episodes',**dict(capacity = config.replay.capacity//10, minlen = config.replay.minlen, maxlen = config.replay.maxlen))
  step = common.Counter(train_replay.stats['total_steps'])
  outputs = [
      common.TerminalOutput(),
      common.JSONLOutput(logdir),
      common.TensorBoardOutput(logdir),
  ]
  logger = common.Logger(step, outputs, multiplier=config.action_repeat)
  metrics = collections.defaultdict(list)

  should_train = common.Every(config.train_every)
  should_log = common.Every(config.log_every)
  should_video_train = common.Every(config.eval_every)
  should_video_eval = common.Every(config.eval_every)
  should_expl = common.Until(config.expl_until // config.action_repeat)

  def make_env(mode):
    suite, task = config.task.split('_', 1)
    if suite == 'dmc':
      env = common.DMC(
          task, config.action_repeat, config.render_size, config.dmc_camera)
      env = common.NormalizeAction(env)
    elif suite == 'atari':
      env = common.Atari(
          task, config.action_repeat, config.render_size,
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
      env = TaxiEnv(size=5,
                    n_passengers=1,
                    exploring_starts=True,
                    terminate_on_goal=True,
                    depot_dropoff_only=False,
                    should_render=True,
                    dimensions=TaxiEnv.dimensions_5x5_to_64x64)
      env = NoiseWrapper(env, sigma=0.01)
      env = ClipWrapper(env, 0.0, 1.0)
      env = TransformWrapper(env, lambda x: x - 0.5)
      env = TimeLimit(env, max_episode_steps=40)
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

  def per_episode(ep, mode):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
    logger.scalar(f'{mode}_return', score)
    logger.scalar(f'{mode}_length', length)
    #pdb.set_trace()
    for key, value in ep.items():
      if re.match(config.log_keys_sum, key):
        logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
      if re.match(config.log_keys_mean, key):
        logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
      if re.match(config.log_keys_max, key):
        logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())


    should = {'train': should_video_train, 'eval': should_video_eval}[mode]
    if should(step):
      for key in config.log_keys_video:
        logger.video(f'{mode}_policy_{key}', ep[key])
    replay = dict(train=train_replay, eval=eval_replay)[mode]
    logger.add(replay.stats, prefix=mode)
    logger.write()

  print('Create envs.')
  num_eval_envs = min(config.envs, config.eval_eps)
  if config.envs_parallel == 'none':
    train_envs = [make_env('train') for _ in range(config.envs)]
    eval_envs = [make_env('eval') for _ in range(num_eval_envs)]
  else:
    make_async_env = lambda mode: common.Async(
        functools.partial(make_env, mode), config.envs_parallel)
    train_envs = [make_async_env('train') for _ in range(config.envs)]
    eval_envs = [make_async_env('eval') for _ in range(eval_envs)]
  act_space = train_envs[0].act_space
  obs_space = train_envs[0].obs_space
  train_driver = common.Driver(train_envs)
  train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
  train_driver.on_step(lambda tran, worker: step.increment())
  train_driver.on_step(train_replay.add_step)
  train_driver.on_reset(train_replay.add_step)
  eval_driver = common.Driver(eval_envs)
  eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
  eval_driver.on_episode(eval_replay.add_episode)

  #pdb.set_trace()
  prefill = max(0, config.prefill - train_replay.stats['total_steps'])
  if prefill:
    print(f'Prefill dataset ({prefill} steps).')
    random_agent = common.RandomAgent(act_space)
    train_driver(random_agent, steps=prefill, episodes=1)
    eval_driver(random_agent, episodes=1)
    train_driver.reset()
    eval_driver.reset()

  #pdb.set_trace()
  print('Create agent.')
  train_dataset = iter(train_replay.dataset(**config.dataset))
  report_dataset = iter(train_replay.dataset(**config.dataset))
  eval_dataset = iter(eval_replay.dataset(**config.dataset))
  agnt = agent.Agent(config, obs_space, act_space, step)
  train_agent = common.CarryOverState(agnt.train)
  train_agent(next(train_dataset))
  if (logdir / 'variables.pkl').exists():
    agnt.load(logdir / 'variables.pkl')
  else:
    print('Pretrain agent.')
    for _ in range(config.pretrain):
      train_agent(next(train_dataset))
  train_policy = lambda *args: agnt.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  eval_policy = lambda *args: agnt.policy(*args, mode='eval')

  def train_step(tran, worker):
    if should_train(step):
      for _ in range(config.train_steps):
        mets = train_agent(next(train_dataset))
        [metrics[key].extend(value) for key, value in mets.items()]

    #pdb.set_trace()
    if should_log(step):

      pred_reward_modes = list(map(lambda value: np.array(value, np.float64), metrics['pred_reward_mode']))
      pred_reward_means = list(map(lambda value: np.array(value, np.float64), metrics['pred_reward_mean']))
      pred_discount_modes = list(map(lambda value: np.array(value, np.float64), metrics['pred_discount_mode']))
      pred_discount_means = list(map(lambda value: np.array(value, np.float64), metrics['pred_discount_mean']))
      actual_rewards = list(map(lambda value: np.array(value, np.float64), metrics['actual_reward']))

      is_timeout = list(map(lambda value: np.array(value, np.int8),  metrics['is_timeout']))
      taxi_rows = list(map(lambda value: np.array(value, np.int8), metrics['taxi_row']))
      taxi_cols = list(map(lambda value: np.array(value, np.int8), metrics['taxi_col']))
      p_rows = list(map(lambda value: np.array(value, np.int8), metrics['p_row']))
      p_cols = list(map(lambda value: np.array(value, np.int8), metrics['p_col']))
      in_taxi = list(map(lambda value: np.array(value, np.int8), metrics['p_in_taxi']))

      for name, values in metrics.items():

        #pred_rewards = [np.array(value, np.float64) for value in values] if name=='pred_reward' else None
        #actual_rewards = [np.array(value, np.float64) for value in values] if name=='actual_reward' else None



        # pdb.set_trace()
        for value in values:

            logger.scalar(name, np.array(value, np.float64).mean())
        metrics[name].clear()
      #pdb.set_trace()
      global reward_tracker

      step_rewards = pd.DataFrame(columns = ['actual_reward', 'is_timeout', 'pred_reward_mode', 'pred_reward_mean', 'pred_discount_mode', 'pred_discount_mean','taxi_row', 'taxi_col', 'p_row', 'p_col','in_taxi'])
      step_rewards['actual_reward'] = actual_rewards
      step_rewards['is_timeout'] = is_timeout
      step_rewards['pred_reward_mode'] = pred_reward_modes
      step_rewards['pred_reward_mean'] = pred_reward_means
      step_rewards['pred_discount_mode'] = pred_discount_modes
      step_rewards['pred_discount_mean'] = pred_discount_means
      step_rewards['taxi_row'] = taxi_rows
      step_rewards['taxi_col'] = taxi_cols
      step_rewards['p_row'] = p_rows
      step_rewards['p_col'] = p_cols
      step_rewards['in_taxi'] = in_taxi

      reward_tracker = reward_tracker.append(step_rewards, ignore_index=True)

      print('Rewards Tracked: ', len(reward_tracker))
      reward_tracker.to_csv('reward_data.csv', index=False)
      logger.add(agnt.report(next(report_dataset)), prefix='train')
      logger.write(fps=True)

  train_driver.on_step(train_step)

  while step < config.steps:
    logger.write()
    print('Start evaluation.')
    logger.add(agnt.report(next(eval_dataset)), prefix='eval')
    
    #pdb.set_trace()
    eval_driver(eval_policy, episodes=config.eval_eps)
    print('Start training.')

    train_driver(train_policy, steps=config.eval_every)
    agnt.save(logdir / 'variables.pkl')


    print('Saving reward')
    #pdb.set_trace()
    #reward_tracker.to_csv(logdir/'reward_data.csv')



  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass


if __name__ == '__main__':
  main()
