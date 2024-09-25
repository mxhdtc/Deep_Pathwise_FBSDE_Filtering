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

project_dir = str(Path(os.getcwd()).resolve())
sys.path.append(project_dir)


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
import os
import json
import argparse



parser = argparse.ArgumentParser(description="Training parameters.")
parser.add_argument('--EPOCH', type=int, help='Training epoch', default=3000)
parser.add_argument('--ID', type=int, help='model id', default=0)
parser.add_argument('--IS_SCORE', type=int, help='Model taining method', default=1)
parser.add_argument('--LAMBDA', type=float, help='Loss function penalty coefficient', default=100.0)
IS_SCORE = parser.parse_args().IS_SCORE
EPOCH = parser.parse_args().EPOCH
MODEL_ID = parser.parse_args().ID
LAMBDA = parser.parse_args().LAMBDA

project_dir = str(Path(os.getcwd()).resolve())
sys.path.append(project_dir)


Target_Tracking_config_path =  os.path.join(os.getcwd(), 'Data/Parameters/Target_Tracking_dim_4_config.json')
with open(Target_Tracking_config_path, 'r') as file:
    Target_Tracking_config = json.load(file)
    
H = np.array([[1.5]], dtype=np.float32) # The observation process drift
sig_dim = Target_Tracking_config['DIM']
obv_dim = Target_Tracking_config['OBV_DIM']
delta_t = Target_Tracking_config['delta_t']
T = Target_Tracking_config['T']
NUM_PATH = Target_Tracking_config['NUM_PATH']
random_seed = Target_Tracking_config['random_seed']
X_obv, Y_obv = Target_Tracking_config['obv_pos']

init_dis = tfp.distributions.MultivariateNormalDiag(loc=[0], scale_diag=[1])

np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# Loading the model drift, diffusion and observation drift
from Data.Target_Tracking import signal_drift, diffusion, obv_drift

model_type = 'Target_Tracking'
config_name = 'Target_Tracking_dim_4_config'
model_name = f"{config_name}"
result_path = os.path.join(os.path.join(os.getcwd(), 'Result/'), model_name)
if not os.path.exists(result_path):
    os.makedirs(result_path)
model_path = os.path.join(os.path.join(os.getcwd(), 'Model/'), model_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)
    
from sklearn.model_selection import train_test_split


# Double_Well_obv_data = np.load("Data/Double_Well/Double_Well_Double_Well_dim_1_gaussian_init_config_obv.npy")
# Double_Well_sig_data = np.load("Data/Double_Well/Double_Well_Double_Well_dim_1_gaussian_init_config_signal.npy")
Target_Tracking_obv_data = np.load("Data/Target_Tracking/Target_Tracking_Target_Tracking_dim_4_config_obv.npy")
Target_Tracking_sig_data = np.load("Data/Target_Tracking/Target_Tracking_Target_Tracking_dim_4_config_signal.npy")
Target_Tracking_sig_data, test_Target_Tracking_sig_data, Target_Tracking_obv_data, test_Target_Tracking_obv_data = train_test_split(Target_Tracking_sig_data, Target_Tracking_obv_data, test_size=0.003, random_state=42)
Target_Tracking_sig_data, verification_Target_Tracking_sig_data, Target_Tracking_obv_data, verification_Target_Tracking_obv_data = train_test_split(Target_Tracking_sig_data, Target_Tracking_obv_data, test_size=0.5, random_state=25)


mu_1, mu_2 = tf.Variable(0.75, dtype=tf.float32), tf.Variable(0.25, dtype=tf.float32)
sigma_1, sigma_2 = tf.Variable(0.4, dtype=tf.float32), tf.Variable(0.3, dtype=tf.float32)
# ---------- build initial distribution and sampling ----------
DIM_1_GMM = tfp.distributions.MixtureSameFamily(
    mixture_distribution=tfp.distributions.Categorical([0.3, 0.7]),
    components_distribution=tfp.distributions.Normal(
        loc=[mu_1, mu_2],
        scale=[sigma_1, sigma_2])
)

if __name__ == '__main__':
    nn_config = ([Target_Tracking_obv_data.shape[1], obv_dim], [128, 64, 64, 32], 16), (sig_dim, [128, 64, 64, 32], 4)


    Target_Tracking_solver = Deep_FBSDE_Filter(delta_t, T, signal_drift, diffusion, obv_drift, sig_dim, obv_dim, DIM_1_GMM, nn_config, build_Zakai_Solver, distribute=True, from_saved_models=False, saved_models = None)

    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=.0001, decay_steps=500, decay_rate=.9)
    train_configuration = Munch(sig_dataset = Target_Tracking_sig_data, obv_dataset=Target_Tracking_obv_data, test_sig_dataset = test_Target_Tracking_sig_data, test_obv_dataset = test_Target_Tracking_obv_data,
                                epoch = EPOCH, batch_size = 64, batch_mc_size = 128, test_batch_mc_size = 128, learning_rate = lr, fbsde=True, LAMBDA=LAMBDA, GAMMA = 0.0)
    Target_Tracking_loss_value =Target_Tracking_solver.dist_train_controlled_sde(config=train_configuration)


    Target_Tracking_solver.Obv_encoder.save(os.path.join(model_path, r'Obv_encoder_Target_Tracking_{ID}.keras'))
    Target_Tracking_solver.Value_function_solver.save(os.path.join(model_path, r'Value_function_solver_Target_Tracking_{ID}.keras'))


    with open(os.path.join(result_path, r'loss_{ID}.npy'), 'wb') as f:
        np.save(f, Target_Tracking_solver.loss_values)
    with open(os.path.join(result_path, r'test_loss_{ID}.npy'), 'wb') as f:
        np.save(f, Target_Tracking_solver.test_loss_values)