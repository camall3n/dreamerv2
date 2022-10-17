import logging
import os
import pathlib
import sys
import warnings

from matplotlib import pyplot as plt
import ruamel.yaml as yaml
import tensorflow as tf
from tensorflow import keras

from dreamerv2 import common
from dreamerv2.agent import Agent
from dreamerv2.train import make_env

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

def setup_log_and_config(exp_path):
    logdir = pathlib.Path(exp_path).expanduser()

    #check if logdir is valid
    if not logdir.exists() or not (logdir / 'saved_models').exists() or len(
            os.listdir((logdir / 'saved_models'))) == 0:
        raise NotImplementedError

    #check if logdir/train_episodes has reference episodes to setup model
    if not (logdir / 'train_episodes').exists() or len(os.listdir(
        (logdir / 'train_episodes'))) == 0:
        raise NotImplementedError

    #store the configs path
    config_path = logdir / 'config.yaml'
    configs = yaml.safe_load(config_path.read_text())

    config = common.Config(configs)

    #tensorflow configuration
    tf.config.run_functions_eagerly(config.jit)
    message = 'No GPU found. To actually train on CPU remove this assert.'
    assert tf.config.experimental.list_physical_devices('GPU'), message
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        from keras import mixed_precision as prec
        prec.set_global_policy(prec.Policy('mixed_float16'))

    #create the DV2 environment
    #
    return logdir, config

class DV2API:
    def __init__(self, logdir, model_file, config, obs_space, act_space):
        #saving config and logdir
        self.config = config
        self.logdir = logdir

        #create agent loading model path
        step = common.Counter(0)

        train_replay = common.Replay(self.logdir / 'train_episodes', **self.config.replay)
        train_dataset = iter(train_replay.dataset(**self.config.dataset))

        self.agnt = Agent(self.config, obs_space, act_space, step)
        initialize_agent = common.CarryOverState(self.agnt.train)
        initialize_agent(next(train_dataset))
        self.agnt.load(self.logdir / 'saved_models' / model_file)

        self.policy = common.RandomAgent(act_space)
        self.state = None

    def dreamerv2_encoder(self, observation_dict):
        # TODO: change observation_dict input to observation

        #change size of dimensions 0 and 1 for tensor
        for k in observation_dict.keys():

            observation_dict[k] = tf.expand_dims(observation_dict[k], axis=0)
            observation_dict[k] = tf.expand_dims(observation_dict[k], axis=0)

            observation_dict[k] = tf.repeat(observation_dict[k], self.config.dataset.batch, axis=0)
            observation_dict[k] = tf.repeat(observation_dict[k],
                                            self.config.dataset.length,
                                            axis=1)

            # TODO: try to get masking working (if we need to keep the [:6, :5] shapes)
            # mask = np.zeros(shape=observation_dict[k].shape)

            # if k is 'image':
            #     mask[0,0,:,:,:] = 1
            #     mask = tf.Variable(mask, dtype=tf.float64)
            # elif k is 'reward':
            #     mask[0] = 1
            #     mask = tf.Variable(mask, dtype= tf.float32)

            # if k is not 'is_terminal':
            #     observation_dict[k] = tf.multiply(observation_dict[k], mask)

        # TODO: Hard code action = 0, is_first = True, is_last = False

        #execute the random agent policy and store action
        action, self.state = self.policy(observation_dict, self.state)

        observation_dict['action'] = tf.repeat(tf.expand_dims(action['action'], axis=1),
                                               self.config.dataset.length,
                                               axis=1)

        observation_dict = self.agnt.wm.preprocess(observation_dict)

        embed = self.agnt.wm.encoder(observation_dict)

        # TODO: check if we can do length-1 trajectories (Priority)
        states, _ = self.agnt.wm.rssm.observe(embed[:6, :5], observation_dict['action'][:6, :5],
                                              observation_dict['is_first'][:6, :5])

        features = self.agnt.wm.rssm.get_feat(states)

        return features

    def dreamerv2_reconstruction(self, features, observation_dict):
        # TODO: reconstruction uses features only, no observation_dict
        reconstructions = []
        ground_truths = []

        for key in self.agnt.wm.heads['decoder'].cnn_keys:
            recon = self.agnt.wm.heads['decoder'](features)[key].mode()[:6]

            model_prediction = tf.cast(recon[:, :5] + 0.5, dtype=tf.float64)
            ground_truth = observation_dict[key][:6] + 0.5

            reconstructions.append(model_prediction[0, 0, :, :, :])
            ground_truths.append(ground_truth[0, 0, :, :, :])

        return reconstructions[0], ground_truths[0]

    def dreamerv2_error(self, reconstructions, ground_truths):
        # errors = []
        # for (recon, ground) in zip(reconstructions, ground_truths):
        #     error = (recon - ground + 1)/2
        #     errors.append(error)

        errors = (reconstructions - ground_truths + 1) / 2
        return errors

def test_api():
    logdir, config = setup_log_and_config('../logs/taxi_1mil_binaryhead_original_seed_01')
    env = make_env(config)
    dv2_api = DV2API(logdir, 'variables_step10001.pkl', config, env.obs_space, env.act_space)
    obs = env.reset()

    print('Image Shape: ', obs['image'].shape)
    if not os.path.exists('./api_test_results'):
        os.mkdir('./api_test_results')

    plt.imshow(obs['image'])
    plt.savefig('./api_test_results/test_gt_image.png')
    plt.cla()
    plt.clf()

    features = dv2_api.dreamerv2_encoder(obs)
    print('Encoded Features Shape: ', features.shape)

    reconstruction, ground_truth = dv2_api.dreamerv2_reconstruction(features, obs)
    print('Reconstructions Shape: ', reconstruction.shape)
    print('Ground Truth Shape: ', ground_truth.shape)

    error = dv2_api.dreamerv2_error(reconstruction, ground_truth)
    print('Error Shape: ', error.shape)

    plt.imshow(reconstruction)
    plt.savefig('./api_test_results/test_reconstruction.png')
    plt.cla()
    plt.clf()

    plt.imshow(ground_truth)
    plt.savefig('./api_test_results/test_ground_truth.png')
    plt.cla()
    plt.clf()

    plt.imshow(error)
    plt.savefig('./api_test_results/test_error.png')
    plt.cla()
    plt.clf()

if __name__ == '__main__':
    test_api()
