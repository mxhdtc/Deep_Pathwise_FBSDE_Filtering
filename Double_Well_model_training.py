import  os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import json
from FBSDE_Filter import Deep_FBSDE_Filter, build_Zakai_Solver
from munch import Munch
from sklearn.model_selection import train_test_split
import argparse



parser = argparse.ArgumentParser(description="Training parameters.")
parser.add_argument('--EPOCH', type=int, help='Training epoch', default=3000)
parser.add_argument('--ID', type=int, help='model id', default=0)
parser.add_argument('--IS_SCORE', type=int, help='Model taining method', default=1)
IS_SCORE = parser.parse_args().IS_SCORE
EPOCH = parser.parse_args().EPOCH
MODEL_ID = parser.parse_args().ID

project_dir = str(Path(os.getcwd()).resolve())
sys.path.append(project_dir)


# Initialize model configuration
config_name = 'Double_Well_dim_1_gaussian_init_config'
Double_Well_path = os.path.join(os.path.join(os.getcwd(), 'Data/Parameter'), f'{config_name}.json')
with open(Double_Well_path, 'r') as file:
    Double_Well_config = json.load(file)
    
H = np.array([[1.5]], dtype=np.float32) # The observation process drift
sig_dim = Double_Well_config['DIM']
obv_dim = H.shape[0]
delta_t = Double_Well_config['delta_t']
T = Double_Well_config['T']
NUM_PATH = Double_Well_config['NUM_PATH']
random_seed = Double_Well_config['random_seed']
init_mu = Double_Well_config['init_mu']
init_sigma = Double_Well_config['init_sigma']

init_dis = tfp.distributions.MultivariateNormalDiag(loc=init_mu, scale_diag=init_sigma)

np.random.seed(random_seed)
tf.random.set_seed(random_seed)

@tf.function
def signal_drift(X, t) -> tf.Tensor:
    """
    :param X: Tensor of shape [batch_size, dim]
    :param t: time step
    :return: drift term at t with shape [batch_size, dim]
    """
    return tf.multiply(.5, tf.multiply(5.0, X) - tf.pow(X, 3))

@tf.function
def obv_drift(X) -> tf.Tensor:
    return tf.einsum('ij, jk->ik', X, tf.transpose(H))

@tf.function
def diffusion(X, t) -> tf.Tensor:
    """
    :param X: Tensor of shape [batch_size, dim]
    :param t: time step
    :return: diffusion term at t
    """
    # return tf.linalg.diag(tf.square(X))
    batch_size, DIM = X.shape[0], X.shape[1]
    return tf.repeat(tf.expand_dims(tf.math.sqrt(2.0) * tf.eye(DIM), axis=0), batch_size, axis=0)

model_type = 'Double_Well'
model_name = f"{config_name}"
result_path = os.path.join(os.path.join(os.getcwd(), 'Result/'), model_name)
if not os.path.exists(result_path):
    os.makedirs(result_path)
model_path = os.path.join(os.path.join(os.getcwd(), 'Model/'), model_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)

init_dis = tfp.distributions.MultivariateNormalDiag(loc=init_mu, scale_diag=init_sigma)
    
# loading trainnig data
Double_Well_obv_data = np.load("Data/Double_Well/Double_Well_Double_Well_dim_1_gaussian_init_config_obv.npy")
Double_Well_sig_data = np.load("Data/Double_Well/Double_Well_Double_Well_dim_1_gaussian_init_config_signal.npy")

Double_Well_sig_data, test_Double_Well_sig_data, Double_Well_obv_data, test_Double_Well_obv_data = train_test_split(Double_Well_sig_data, Double_Well_obv_data, test_size=0.003, random_state=42)

# Parameterize the solver neural networks model
nn_config = ([Double_Well_sig_data.shape[1], 1], [128, 128, 64, 32], 4), (1, [16, 32, 64], 1)

# saved_models = None
saved_models = tf.keras.models.load_model(os.path.join(model_path, 'Value_function_solver_reformulate_6.keras')), tf.keras.models.load_model(os.path.join(model_path, 'Obv_encoder_reformulate_6.keras'))

Double_Well_solver = Deep_FBSDE_Filter(delta_t, T, signal_drift, diffusion, obv_drift, sig_dim, obv_dim, init_dis, nn_config, build_Zakai_Solver, distribute=True, from_saved_models=True, saved_models = saved_models, is_score = IS_SCORE)

print(Double_Well_solver.Obv_encoder.summary())

# Configure the training procedure
lr = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-5, decay_steps=500, decay_rate=.9)
train_configuration = Munch(sig_dataset = Double_Well_sig_data, obv_dataset=Double_Well_obv_data, test_sig_dataset = test_Double_Well_sig_data, test_obv_dataset=test_Double_Well_obv_data,
                            epoch = EPOCH, batch_size = 32, batch_mc_size = 32, test_batch_mc_size = 32, learning_rate = lr, fbsde=True, LAMBDA=1.0, GAMMA = 0.0)

# Train and save the model
Double_Well_loss_value = Double_Well_solver.dist_train_controlled_sde(config=train_configuration)

with open(os.path.join(result_path, 'loss_{}.npy'.format(MODEL_ID)), 'wb') as f:
    np.save(f, Double_Well_solver.loss_values)
with open(os.path.join(result_path, 'test_loss_{}.npy'.format(MODEL_ID)), 'wb') as f:
    np.save(f, Double_Well_solver.test_loss_values)

Double_Well_solver.Obv_encoder.save(os.path.join(model_path, 'Obv_encoder_reformulate_{}.keras'.format(MODEL_ID))) # reformulate means that we use Ito's formula to set the terminal cost to be zero
Double_Well_solver.Value_function_solver.save(os.path.join(model_path, 'Value_function_solver_reformulate_{}.keras'.format(MODEL_ID)))