import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
import numpy as np
from munch import Munch
import tensorflow_recommenders as tfrs
from tensorflow.keras.models import clone_model
import time
from tqdm import tqdm
from tabulate import tabulate

LOSS_CLIP = 1.0


def build_Zakai_Solver(config):
    encoder_config, decoder_config = config
    encoder_input_dim, encoder_hidden_layers, encoder_output_dim = encoder_config
    decoder_input_dim, decoder_hidden_layers, decoder_output_dim = decoder_config
    decoder_input_dim += encoder_output_dim + 1  # the input of the decoder is the concatenation of the inpute signal
    # state, time and hidden representation of observation state.
    encoder_model_layers = [
        tf.keras.layers.InputLayer(input_shape=(encoder_input_dim[0], encoder_input_dim[1])),
    ]
    for units in encoder_hidden_layers:
        encoder_model_layers.append(
            # tf.keras.layers.Bidirectional(
            # layer = tf.keras.layers.LSTM(units, return_sequences=True),
            # backward_layer = tf.keras.layers.LSTM(units, return_sequences=True, go_backwards=True),
            # )
            tf.keras.layers.LSTM(units, return_sequences=True, go_backwards=True)
        )
    encoder_model_layers.append(tf.keras.layers.LSTM(encoder_output_dim, return_sequences=True, go_backwards=True))
    decoder_model_layers = [
        tf.keras.layers.InputLayer(input_shape=(decoder_input_dim,))
    ]
    # Add Dense and BatchNormalization layers for each hidden layer
    for units in decoder_hidden_layers:
        # decoder_model_layers.append(tf.keras.layers.BatchNormalization(
        #     momentum=0.99,
        #     epsilon=1e-6,
        #     beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
        #     gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
        # ))  # Add BatchNormalization layer
        # decoder_model_layers.append(tf.keras.layers.Dropout(0.2))
        decoder_model_layers.append(
            tf.keras.layers.Dense(units=units, activation='relu',
                                  kernel_initializer=tf.keras.initializers.LecunNormal(seed=np.random.randint(1e9)))
        )
        # decoder_model_layers.append(tf.keras.layers.Activation('relu'))  # Add Activation layer after BatchNormalization

        # Add final Dense layer without BatchNormalization
    decoder_model_layers.append(
        tf.keras.layers.Dense(units=decoder_output_dim, activation='linear',
                              kernel_initializer=tf.keras.initializers.LecunNormal(seed=np.random.randint(1e9)))
    )

    return tf.keras.Sequential(decoder_model_layers), tf.keras.Sequential(encoder_model_layers)


class Deep_FBSDE_Filter(object):
    def __init__(self, delta_t, T, drift, diffusion, obv_drift, sig_dim, obv_dim, init_dis, nn_config, build_neural_networks,
                 distribute=None, is_score=True, from_saved_models=False, saved_models=None):
        """
        The model for solving the stochastic control problem by neural networks,
        which can also provide the samplings of the optimal controlled SDE with visualization methods.

        Parameters
        ----------
        delta_t :
            The gap of each time step
        T :
            Terminal time of the horizon
        drift :
        sampling_drift:
        diffusion
        DIM
        init_dis
        is_score : If true, the neural networks will generate the score function
            (gradient of the log-probability with respect to x), otherwise, they will
            generate the log-probability function.
        """
        # super(Reverse_Fokker_Planck_Solver, self).__init__()
        # self.DIM = DIM
        self.test_loss_values = None
        self.loss_values = None
        if saved_models is None:
            saved_models = [None, None]
        self.sig_dim = sig_dim
        self.obv_dim = obv_dim
        self.delta_t = delta_t
        self.T = T
        self.init_dis = init_dis
        self.log_prob_init = self.init_dis.log_prob
        self.drift = drift
        self.diffusion = diffusion
        self.obv_drift = obv_drift
        self.nn_config = nn_config
        self.build_neural_networks = build_neural_networks
        # self.hidden_layers = hidden_layers
        # self.output_dim = output_dim
        self.is_score = is_score

        def build_models():
            if from_saved_models:
                self.Value_function_solver, self.Obv_encoder = clone_model(saved_models[0]), clone_model(saved_models[1])
                self.Value_function_solver.set_weights(saved_models[0].get_weights())
                self.Obv_encoder.set_weights(saved_models[1].get_weights())
            else:
                self.Value_function_solver, self.Obv_encoder = self.build_neural_networks(self.nn_config)
        # ---------- MLP Model for learning the PDE solution ----------
        if distribute is None:
            self.strategy = None
            build_models()
            # self.Fokker_Planck_Solver = self.build_Fokker_Planck_Solver(self.DIM + 1, hidden_layers, output_dim)
            self.controlled_drift = self.build_controlled_drift(self.drift, self.diffusion, self.Value_function_solver)
        else:
            self.strategy = tf.distribute.MirroredStrategy()
            print("Number of GPUs Available: {}".format(self.strategy.num_replicas_in_sync))
            with self.strategy.scope():
                build_models()
                # self.Value_function_solver, self.Obv_encoder = self.build_neural_networks(self.nn_config)
                # self.Fokker_Planck_Solver = self.build_Fokker_Planck_Solver(self.DIM + 1, hidden_layers, output_dim)
                self.controlled_drift = self.build_controlled_drift(self.drift, self.diffusion,
                                                                    self.Value_function_solver)

    # ---------- define the controlled drift ----------

    def build_controlled_drift(self, drift, diffusion, controller):
        """
        Parameters
        ----------
        drift
        diffusion
        controller

        Returns
        -------

        """

        # """
        # :param X: Tensor of shape [batch_size, dim]
        # :param t: time step
        # :param controller: Using a neural network modeled the \nabla log U(X_t, t)
        # with inpute [X, t]
        # :return: drift term at t with shape [batch_size, dim]
        # """
        if self.is_score:
            def controlled_drift(X, t, y_hidden_t, y_t):
                batch_size, DIM = X.shape[0], X.shape[1]
                t = tf.repeat(tf.expand_dims([t], axis=0), batch_size, axis=0)  # transfer t to batch size
                # X, t = tf.Variable(X), tf.Variable(t)
                with tf.GradientTape() as tape:
                    tape.watch(X)
                    nabla_log_v = controller(tf.concat([X, t, y_hidden_t], axis=-1))
                    h = self.obv_drift(X)
                    nabla_h = tape.batch_jacobian(h, X)
                    y_dot_nabla_h = tf.einsum('ij,ijk->ik', y_t, nabla_h)
                del tape
                # log_u = controller(tf.concat([X, t], axis=-1))
                # nabla_log_u = tape.gradient(log_u, X)  # get the score function
                sigma_T = tf.transpose(diffusion(X, t), perm=[0, 2, 1])
                # print(drift(X, t).shape)
                # print(nabla_log_u.shape)
                # print(sigma_T.shape)
                ###
                return -1.0 * tf.einsum('ijk, ik->ij', sigma_T, nabla_log_v) - 1.0 * tf.einsum('ijk,ik->ij', sigma_T, y_dot_nabla_h)
        else:
            def controlled_drift(X, t, y_hidden_t, y_t):
                batch_size, DIM = X.shape[0], X.shape[1]
                t = tf.repeat(tf.expand_dims([t], axis=0), batch_size, axis=0)  # transfer t to batch size
                # X, t = tf.Variable(X), tf.Variable(t)
                with tf.GradientTape() as tape:
                    tape.watch(X)
                    # log_u = -1.0 * tf.math.log(controller(tf.concat([X, t], axis=-1)))
                    log_v = controller(tf.concat([X, t, y_hidden_t], axis=-1))
                    # log_u += tf.reshape(-1.0 * self.log_prob_init(X), log_u.shape)
                    # log_u = controller(tf.concat([X, t], axis=-1))
                    nabla_log_v = tape.gradient(log_v, X)  # get the score function
                    h = self.obv_drift(X)
                    nabla_h = tape.batch_jacobian(h, X)
                    y_dot_nabla_h = tf.einsum('ij,ijk->ik', y_t, nabla_h)
                del tape
                sigma_T = tf.transpose(diffusion(X, t), perm=[0, 2, 1])
                # print(drift(X, t).shape)
                # print(nabla_log_u.shape)
                # print(sigma_T.shape)
                return -1.0 * tf.einsum('ijk, ik->ij', sigma_T, nabla_log_v) - 1.0 * tf.einsum('ijk,ik->ij', sigma_T,
                                                                                               y_dot_nabla_h)
            # return -1.0 * tf.einsum('ijk, ik->ij', sigma_T, log_u)

        # def controlled_drift(X, t):
        #     batch_size, DIM = X.shape[0], X.shape[1]
        #     t = tf.repeat(tf.expand_dims([t], axis=0), batch_size, axis=0)  # transfer t to batch size
        #     # X, t = tf.Variable(X), tf.Variable(t)
        #     u = controller(tf.concat([X, self.T - t], axis=-1))
        #     sigma_T = tf.transpose(diffusion(X, self.T - t), perm=[0, 2, 1])
        #     # print(drift(X, t).shape)
        #     # print(nabla_log_u.shape)
        #     # print(sigma_T.shape)
        #     return tf.einsum('ijk, ik->ij', sigma_T, u)
        return controlled_drift

    def sde_sampling(self, init_state, config):
        """
        Sampling sde through Euler maruyama scheme

        Parameters
        ----------
        - init_state: shape (M, 1, d)
        - config: Munch object, contains
            - T: Sampling period
            - reset_configuration: If False, use self object as sampling parameters. Otherwise, will reset
                new configurations
        Returns
        -------
        Euler Maruyama numerical results with shape (M, N, d)
        """

        # ---------- Set sampling configuration ----------
        def _get_configuration(config):
            if config.reset_configuration:
                return config.T, config.delta_t, config.drift, config.diffusion
            else:
                return self.T, self.delta_t, self.drift, self.diffusion

        batch_size, DIM = init_state.shape[0], init_state.shape[1]
        T, delta_t, drift, diffusion = _get_configuration(config)
        # if config.reset_configuration:
        #     T = config.T
        #     delta_t = config.delta_t
        #     drift = config.drift
        #     diffusion = config.diffusion
        # else:
        #     T = self.T
        #     delta_t = self.delta_t
        #     drift = self.drift
        #     diffusion = self.diffusion
        brownian_motion = tfp.distributions.Normal(loc=0.0, scale=1.0).sample(
            sample_shape=[batch_size, int(np.ceil(T / delta_t)), DIM])

        # ---------- define one step EM sampling ----------
        def one_step_em(current_state, gaussian_noise):
            t = current_state[0][-1]  # extract the current time
            x = current_state[:, :-1]
            iterated_state = x + delta_t * drift(x, t) + tf.sqrt(delta_t) * tf.einsum('ijk, ik->ij', diffusion(x, t),
                                                                                      gaussian_noise)
            return tf.concat([iterated_state, current_state[:, -1:] + delta_t], axis=-1)

        trajectories = tf.scan(one_step_em, tf.transpose(brownian_motion, perm=[1, 0, 2]),
                               initializer=tf.concat([init_state, tf.zeros([batch_size, 1])], axis=-1))
        return tf.concat([tf.expand_dims(init_state, axis=1), tf.transpose(trajectories[:, :, :-1], perm=[1, 0, 2])],
                         axis=1), brownian_motion

    # @tf.function
    def controlled_sde_sampling(self, init_state, obv_trajectoies, config):
        """
        Sampling a controlled SDE with Euler Maruyama scheme
        Parameters
        ----------
        init_state: size (M, N, d_sig), M stands for batch size, N stands for time steps, d_sig stands for the dimension of the signal state,
        obv_trajectories: size (M, N, d_obv), M stands for batch size, N stands for time steps, d_obv stands for the dimension of the observation state.

        Returns
        -------

        """

        # ---------- Set sampling configuration ----------
        def _get_configuration(config):
            if config.reset_configuration:
                return (config.T, config.delta_t, config.drift, config.controlled_drift, config.diffusion,
                        config.controller)
            else:
                return (self.T, self.delta_t, self.drift, self.controlled_drift, self.diffusion, self.
                        Value_function_solver)

        batch_size, DIM = init_state.shape[0], init_state.shape[1]
        T, delta_t, drift, controlled_drift, diffusion, controller = _get_configuration(config)

        current_time = int(time.time())
        tf.random.set_seed(current_time)
        brownian_motion = tfp.distributions.Normal(loc=0.0, scale=1.0).sample(
            sample_shape=[batch_size, int(np.ceil(T / delta_t)), DIM])

        # ----------Define One step EM sampling----------
        num_steps = int(np.ceil(T / delta_t)) + 1
        time_steps = tf.linspace(0.0, (num_steps - 2), num_steps - 1)
        y_hidden = self.Obv_encoder(obv_trajectoies)[:, :-1, :]
        # y_hidden = self.Obv_encoder(obv_trajectoies[:, 1:, :] - obv_trajectoies[:, :-1, :])
        y = obv_trajectoies[:, :-1, :]
        y_T = obv_trajectoies[:, -1, :]
        def one_step_controlled_em(current_state, inputs):
            y_hidden_t, y_t, gaussian_noise, t = inputs  # extract the current time
            x, t = current_state, t * delta_t
            sigma = diffusion(x, t)
            # iterated_state = x + delta_t * controlled_drift(x, t, drift=drift, diffusion=diffusion,
            #                                                 controller=controller) + tf.sqrt(delta_t) * tf.einsum(
            #     'ijk, ik->ij', diffusion(x, t), gaussian_noise)
            # change the Hamiltonian
            iterated_state = (x + delta_t * (drift(x, t)
                                             + tf.einsum('ijk, ik->ij', sigma, controlled_drift(x, t, y_hidden_t, y_t - y_T))
                                             )
                              + tf.sqrt(delta_t) * tf.einsum('ijk, ik->ij', sigma, gaussian_noise))
            # controlled_drift = controlled_drift[1:]
            return iterated_state

        trajectories = tf.scan(one_step_controlled_em,
                               (tf.transpose(y_hidden, perm=[1, 0, 2]), tf.transpose(y, perm=[1, 0, 2]),
                                tf.transpose(brownian_motion, perm=[1, 0, 2]), time_steps),
                               initializer=init_state)
        return tf.concat([tf.expand_dims(init_state, axis=1), tf.transpose(trajectories, perm=[1, 0, 2])],
                         axis=1), brownian_motion

    def dist_train_controlled_sde(self, config):
        # ---------- Training Configuration Initialization----------
        sig_dataset, obv_dataset, test_sig_dataset, test_obv_dataset,epoch, batch_size, batch_mc_size, test_batch_mc_size, learning_rate, fbsde, LAMBDA, GAMMA \
            = (config.sig_dataset, config.obv_dataset, config.test_sig_dataset, config.test_obv_dataset, config.epoch, config.batch_size, config.batch_mc_size,
               config.test_batch_mc_size, config.learning_rate, config.fbsde, config.LAMBDA, config.GAMMA)
        # learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=.01, decay_steps=100000, decay_rate=.9)

        with ((self.strategy.scope())):
            # optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)
            value_function_optimizer = tf.keras.optimizers.Adam(learning_rate)
            encoder_optimizer = tf.keras.optimizers.Adam(learning_rate)
            # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            # optimizer = tfrs.experimental.optimizers.ClippyAdagrad(learning_rate=learning_rate)
            # optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

            # ---------- Define the ELBO of batch of trajectories ----------

            def fbsde_trajectory_running_loss(trajectories, obv_trajectoies, brownian_motion):
                """
                Calculates the Monte Carlo estimate of the cost function J(u, x) for given trajectories.
                Parameters
                ----------
                - trajectories: Tensor of shape [batch_size, num_steps, dim], representing sampled trajectories.

                Returns
                -------
                - The Monte Carlo estimate of J(u, x).
                """
                batch_size, num_steps, DIM = trajectories.shape

                # Prepare time steps tensor
                time_steps = tf.linspace(0.0, (num_steps - 2), num_steps - 1)
                y_hidden = self.Obv_encoder(obv_trajectoies)[:, :-1, :]
                # y_hidden = self.Obv_encoder(obv_trajectoies[:, 1:, :] - obv_trajectoies[:, :-1, :])
                y_T = obv_trajectoies[:, -1, :]
                def accumulate_internal_cost(accumulate_internal, inputs):
                    x, t, dw, y_hidden_t, y_t = inputs
                    y_t = y_t - y_T
                    t = t * self.delta_t
                    sigma_T = tf.transpose(self.diffusion(x, t), perm=[0, 2, 1])
                    sigma = self.diffusion(x, t)
                    # x, t = tf.Variable(x), tf.Variable(t * self.delta_t)
                    with tf.GradientTape() as tape_1:
                        tape_1.watch(x)
                        with tf.GradientTape() as tape_2:
                            tape_2.watch(x)
                        drift_x_t = self.drift(x, t)
                        h = self.obv_drift(x)
                        nabla_h = tape_1.batch_jacobian(h, x)
                    hessian_h = tape_2.batch_jacobian(nabla_h, x)
                    controlled_drift = self.controlled_drift(x, t, y_hidden_t, y_t)
                    controlled_drift_square = tf.reduce_sum(tf.square(controlled_drift), axis=-1)
                    h_square = tf.reduce_sum(tf.square(h), axis=-1)
                    y_dot_nabla_h = tf.einsum('ij,ijk->ik', y_t, nabla_h)
                    y_dot_hessian_h = tf.einsum('ij,ijkl->ikl', y_t, hessian_h)
                    integral_step = (tf.einsum('ij,ij->i',
                                               drift_x_t + tf.einsum('ijk,ik->ij', sigma, controlled_drift),
                                               y_dot_nabla_h)
                                     + 0.5 * tf.linalg.trace(
                                tf.einsum('ijl,ilm->ijm', tf.einsum('ijk,ikl->ijl', sigma_T, y_dot_hessian_h), sigma))
                                     + 0.5 * controlled_drift_square
                                     + 0.5 * h_square
                                     ) * self.delta_t
                    t_batch = tf.repeat(tf.expand_dims([t], axis=0), batch_size, axis=0)
                    martingale_increment = tf.sqrt(self.delta_t) * \
                                           tf.einsum('ij,ij->i', tf.einsum('ijk, ik->ij', sigma_T,
                                                       self.Value_function_solver(
                                                       tf.concat([x, t_batch, y_hidden_t], axis=-1))), dw)
                    # print(integral_step.shape)
                    # print(nabla_dot_drift.shape)
                    # print(controlled_drift_square.shape)
                    return accumulate_internal + integral_step - martingale_increment

                # X_T = trajectories[:, -1, :]

                # Initial accumulator value for tf.scan, set to 0
                initial_accumulator = tf.zeros([batch_size], dtype=tf.float32)

                # Prepare inputs for tf.scan: (X_t, t, dw)
                trajectory_inputs = (tf.transpose(trajectories[:, :-1, :], perm=[1, 0, 2]),
                                     time_steps, tf.transpose(brownian_motion, perm=[1, 0, 2]),
                                     tf.transpose(y_hidden, perm=[1, 0, 2]),
                                     tf.transpose(obv_trajectoies[:, :-1, :], perm=[1, 0, 2]),)
                integral = tf.scan(accumulate_internal_cost, trajectory_inputs, initializer=initial_accumulator)
                integral = integral[-1]
                # return tf.reduce_mean(integral - self.init_dis.log_prob(X_T))
                # print(X_T.shape, integral.shape)
                loss_results = 1.0 * integral
                # - self.init_dis.log_prob(X_T)[:, 0]
                # print(loss_results.shape)
                return loss_results

            def fbsde_trajectory_terminal_loss(trajectories, obv_trajectoies, brownian_motion):
                """
                Calculates the Monte Carlo estimate of the cost function J(u, x) for given trajectories.
                Parameters
                ----------
                - trajectories: Tensor of shape [batch_size, num_steps, dim], representing sampled trajectories.

                Returns
                -------
                - The Monte Carlo estimate of J(u, x).
                """
                # batch_size, num_steps, DIM = trajectories.shape

                # Prepare time steps tensor
                # time_steps = tf.linspace(0.0, (num_steps - 2), num_steps - 1)

                X_T = trajectories[:, -1, :]
                y_T = obv_trajectoies[:, -1, :]
                loss_results = tf.einsum('ij,ij->i', y_T, -1.0 * self.obv_drift(X_T))
                # print(loss_results.shape)
                # return loss_results
                return tf.zeros_like(loss_results)

        def train_step(inputs):
            init_samples, observation_trajectories = inputs
            num_steps = observation_trajectories.shape[2]
            init_samples = tf.reshape(init_samples, (batch_size * batch_mc_size, self.sig_dim))
            observation_trajectories = tf.reshape(observation_trajectories, (batch_size * batch_mc_size, num_steps, self.obv_dim))
            configuration = Munch(reset_configuration=None)

            with (tf.GradientTape() as tape):
                controlled_samples, dw_2 = self.controlled_sde_sampling(
                    # tf.zeros([batch_size * batch_mc_size, self.DIM]),
                    init_samples,
                    observation_trajectories,
                    configuration)
                # controlled_samples = tf.stop_gradient(controlled_samples)
                # dw_2 = tf.stop_gradient(dw_2)
                # tf.stop_gradient(reversed_samples)
                # tf.stop_gradient(dw_2)
                # reversed_samples_static, dw_3 = self.controlled_sde_sampling(
                #     # tf.zeros([batch_size * batch_mc_size, self.DIM]),
                #     init_samples,
                #     configuration)
                # tf.stop_gradient(reversed_samples_static)
                # tf.stop_gradient(dw_3)
                terminal_loss = tf.reshape(fbsde_trajectory_terminal_loss(controlled_samples, observation_trajectories, dw_2),
                                           [batch_size, batch_mc_size])
                running_loss = tf.reshape(fbsde_trajectory_running_loss(controlled_samples, observation_trajectories, dw_2),
                                          [batch_size, batch_mc_size])
                V_0 = tf.identity(terminal_loss + running_loss)
                terminal_loss = tf.reduce_mean(terminal_loss, axis=1)
                running_loss = tf.reduce_mean(running_loss, axis=1)
                # print(terminal_loss.shape, running_loss.shape)
                # runnning_loss = fbsde_trajectory_running_loss(reversed_samples_static, dw_3)
                # loss = tf.where(fbsde, fbsde_trajectory_loss(reversed_samples, dw_2),
                #                 _trajectory_loss(reversed_samples))
                # loss = _trajectory_loss(trajectories=reversed_samples)
                value_loss = tf.nn.compute_average_loss(terminal_loss + running_loss)

                V_0 = tf.reshape(V_0, [batch_size, batch_mc_size])
                expected_V_0 = tf.reduce_mean(V_0, axis=1, keepdims=True)
                measurable_loss = tf.square(V_0 - expected_V_0)
                measurable_loss = tf.reduce_mean(measurable_loss, axis=1)
                # measurable_loss = tf.reshape(measurable_loss, [batch_size])
                measurable_loss = tf.nn.compute_average_loss(measurable_loss)
                # boundary_samples = tf.reshape(reversed_samples_static[:, -1, :],
                # [batch_size * batch_mc_size, self.DIM])
                # boundary_samples = tf.reshape(boundary_samples, [batch_size * batch_mc_size, self.DIM])
                # t = tf.repeat(tf.expand_dims([self.T], axis=0), reversed_samples.shape[0], axis=0)
                # boundary_loss = tf.nn.compute_average_loss(
                #     tf.reduce_mean(
                #         tf.reshape(
                #             tf.square(
                #                 tf.reshape(self.Solver(
                #                     tf.concat([boundary_samples, t], axis=-1)), [batch_size * batch_mc_size])
                #                 # - tf.reshape(self.init_dis.log_prob(boundary_samples), [batch_size * batch_mc_size])
                #             ), [batch_size, batch_mc_size]), axis=1)
                # )

                loss = value_loss + LAMBDA * measurable_loss
                # + GAMMA * boundary_loss
                # loss = boundary_loss

                loss = tf.where(loss < LOSS_CLIP, loss,
                                2 * LOSS_CLIP * loss - LOSS_CLIP ** 2)
                trainable_vars = self.Value_function_solver.trainable_variables + self.Obv_encoder.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)
                value_function_solver_gradients = gradients[:len(self.Value_function_solver.trainable_variables)]
                encoder_gradients = gradients[len(self.Value_function_solver.trainable_variables):]

# value_function_solver_gradients = tape.gradient(loss, self.Value_function_solver.trainable_variables)
                # encoder_gradients = tape.gradient(loss, self.Obv_encoder.trainable_variables)
                # gradients = tape.gradient(measurable_loss, self.Fokker_Planck_Solver.trainable_variables)

                value_function_optimizer.apply_gradients(zip(value_function_solver_gradients, self.Value_function_solver.trainable_variables))
                encoder_optimizer.apply_gradients(zip(encoder_gradients, self.Obv_encoder.trainable_variables))
                loss = tf.where(loss < LOSS_CLIP, loss,
                                (loss + LOSS_CLIP ** 2) / (2 * LOSS_CLIP))
                terminal_loss = tf.nn.compute_average_loss(terminal_loss)
                running_loss = tf.nn.compute_average_loss(running_loss)

            return loss, value_loss, measurable_loss, running_loss, terminal_loss

        def test_step(inputs):
            init_samples, observation_trajectories = inputs
            num_steps = observation_trajectories.shape[2]
            batch_size = observation_trajectories.shape[0]
            init_samples = tf.reshape(init_samples, (batch_size * test_batch_mc_size, self.sig_dim))
            observation_trajectories = tf.reshape(observation_trajectories, (batch_size * test_batch_mc_size, num_steps, self.obv_dim))
            configuration = Munch(reset_configuration=None)

            controlled_samples, dw_2 = self.controlled_sde_sampling(
                init_samples,
                observation_trajectories,
                configuration)
            terminal_loss = tf.reshape(fbsde_trajectory_terminal_loss(controlled_samples, observation_trajectories, dw_2),
                                       [batch_size, test_batch_mc_size])
            running_loss = tf.reshape(fbsde_trajectory_running_loss(controlled_samples, observation_trajectories, dw_2),
                                      [batch_size, test_batch_mc_size])
            V_0 = tf.identity(terminal_loss + running_loss)

            terminal_loss = tf.reduce_mean(terminal_loss, axis=1)
            #
            running_loss = tf.reduce_mean(running_loss, axis=1)
            value_loss = tf.nn.compute_average_loss(terminal_loss + running_loss)
            V_0 = tf.reshape(V_0, [batch_size, test_batch_mc_size])
            expected_V_0 = tf.reduce_mean(V_0, axis=1, keepdims=True)
            measurable_loss = tf.square(V_0 - expected_V_0)
            measurable_loss = tf.reduce_mean(measurable_loss, axis=1)
            measurable_loss = tf.nn.compute_average_loss(measurable_loss)
            test_loss = value_loss + LAMBDA * measurable_loss
            return test_loss, value_loss, measurable_loss


#         def train_step(inputs):
#             init_samples, observation_trajectories = inputs
#             num_steps = observation_trajectories.shape[2]

#             split_init_samples = tf.split(init_samples, repeat_number, axis=0)
#             split_observation_trajectories = tf.split(observation_trajectories, repeat_number, axis=0)

#             accumulated_gradients = [tf.zeros_like(var) for var in self.Value_function_solver.trainable_variables + self.Obv_encoder.trainable_variables]

#             total_loss = 0
#             total_value_loss = 0
#             total_measurable_loss = 0
#             total_running_loss = 0
#             total_terminal_loss = 0

#             for i in range(repeat_number):
#                 init_samples_split = tf.reshape(split_init_samples[i], (batch_size * batch_mc_size // repeat_number, self.sig_dim))
#                 observation_trajectories_split = tf.reshape(split_observation_trajectories[i], (batch_size * batch_mc_size // repeat_number, num_steps, self.obv_dim))
#                 configuration = Munch(reset_configuration=None)

#                 with tf.GradientTape() as tape:
#                     controlled_samples, dw_2 = self.controlled_sde_sampling(
#                         init_samples_split,
#                         observation_trajectories_split,
#                         configuration)
#                     terminal_loss = tf.reshape(fbsde_trajectory_terminal_loss(controlled_samples, observation_trajectories_split, dw_2),
#                                                [batch_size // repeat_number, batch_mc_size])
#                     running_loss = tf.reshape(fbsde_trajectory_running_loss(controlled_samples, observation_trajectories_split, dw_2),
#                                               [batch_size // repeat_number, batch_mc_size])
#                     V_0 = tf.identity(terminal_loss + running_loss)
#                     terminal_loss = tf.reduce_mean(terminal_loss, axis=1)
#                     running_loss = tf.reduce_mean(running_loss, axis=1)
#                     value_loss = tf.nn.compute_average_loss(terminal_loss + running_loss)

#                     V_0 = tf.reshape(V_0, [batch_size // repeat_number, batch_mc_size])
#                     expected_V_0 = tf.reduce_mean(V_0, axis=1, keepdims=True)
#                     measurable_loss = tf.square(V_0 - expected_V_0)
#                     measurable_loss = tf.reduce_mean(measurable_loss, axis=1)
#                     measurable_loss = tf.nn.compute_average_loss(measurable_loss)

#                     loss = value_loss + LAMBDA * measurable_loss
#                     loss = tf.where(loss < LOSS_CLIP, loss, 2 * LOSS_CLIP * loss - LOSS_CLIP ** 2)

#                     trainable_vars = self.Value_function_solver.trainable_variables + self.Obv_encoder.trainable_variables
#                     gradients = tape.gradient(loss, trainable_vars)

#                     accumulated_gradients = [accum_grad + grad for accum_grad, grad in zip(accumulated_gradients, gradients)]

#                     total_loss += loss
#                     total_value_loss += value_loss
#                     total_measurable_loss += measurable_loss
#                     total_running_loss += running_loss
#                     total_terminal_loss += terminal_loss

#             accumulated_gradients = [grad / repeat_number for grad in accumulated_gradients]

#             value_function_solver_gradients = accumulated_gradients[:len(self.Value_function_solver.trainable_variables)]
#             encoder_gradients = accumulated_gradients[len(self.Value_function_solver.trainable_variables):]

#             value_function_optimizer.apply_gradients(zip(value_function_solver_gradients, self.Value_function_solver.trainable_variables))
#             encoder_optimizer.apply_gradients(zip(encoder_gradients, self.Obv_encoder.trainable_variables))

#             total_loss /= repeat_number
#             total_value_loss /= repeat_number
#             total_measurable_loss /= repeat_number
#             total_running_loss /= repeat_number
#             total_terminal_loss /= repeat_number

#             return total_loss, total_value_loss, total_measurable_loss, total_running_loss, total_terminal_loss


        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses = self.strategy.run(train_step, args=(dataset_inputs,))
            return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                        axis=None)

        @tf.function
        def distributed_test_step(dataset_inputs):
            per_replica_test_losses = self.strategy.run(test_step, args=(dataset_inputs,))
            return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_test_losses,
                                        axis=None)
        
        
        def print_loss_table(epoch, loss, test_loss):
            headers = ["Epoch", "Loss", "Value Loss", "Measurable Loss", "Running Loss", "Terminal Loss", "Test Value Loss", "Test Measurable Loss"]
            row = [
                epoch + 1, 
                loss[0].numpy(), 
                loss[1].numpy(), 
                loss[2].numpy(), 
                loss[3].numpy(), 
                loss[4].numpy(), 
                test_loss[1].numpy(), 
                test_loss[2].numpy()
            ]
            table = tabulate([row], headers=headers, tablefmt="grid")
            print(table)

            
        # ---------- Training Loops ----------
        loss_values = []
        test_loss_values = []
        configuration = Munch(reset_configuration=None)
        print("start training procedure!")
        dataset_size = sig_dataset.shape[0]
        for i in range(epoch):
            total_indices = tf.range(start=0, limit=dataset_size, dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(total_indices)
            batch_indices = shuffled_indices[:batch_size * self.strategy.num_replicas_in_sync]
            init_samples = tf.gather(sig_dataset, batch_indices)[:, 0, :]
            observation_trajectories = tf.gather(obv_dataset, batch_indices)
            # forward_samples, dw = self.sde_sampling(
            #     tf.reshape(self.init_dis.sample(
            #         sample_shape=[batch_size * self.strategy.num_replicas_in_sync]),
            #         (batch_size * self.strategy.num_replicas_in_sync, self.sig_dim))
            #     , configuration)
            # forward_samples = tf.expand_dims(forward_samples[:, -1, :], axis=1)
            # forward_samples = tf.repeat(forward_samples, batch_mc_size, axis=1)
            init_samples, observation_trajectories = tf.expand_dims(init_samples, axis=1), tf.expand_dims(observation_trajectories, axis=1)
            init_samples, observation_trajectories = tf.repeat(init_samples, batch_mc_size, axis=1), tf.repeat(observation_trajectories, batch_mc_size, axis=1)
            test_init_samples, test_observation_trajectories = tf.expand_dims(test_sig_dataset[:, 0, :], axis=1), tf.expand_dims(test_obv_dataset, axis=1)
            test_init_samples, test_observation_trajectories = tf.repeat(test_init_samples, test_batch_mc_size, axis=1), tf.repeat(test_observation_trajectories, test_batch_mc_size, axis=1)
            # static_boundary_samples = tf.reshape(self.init_dis.sample(
            #     sample_shape=[batch_size * self.strategy.num_replicas_in_sync * batch_mc_size]),
            #     (batch_size * self.strategy.num_replicas_in_sync, batch_mc_size, self.DIM))

            print("Successfully sampling training data!")
            data_set = tf.data.Dataset.from_tensor_slices((init_samples, observation_trajectories)).batch(
                batch_size * self.strategy.num_replicas_in_sync)
            dis_data_set = self.strategy.experimental_distribute_dataset(data_set)
            test_data_set = tf.data.Dataset.from_tensor_slices((test_init_samples, test_observation_trajectories)).batch(
                test_obv_dataset.shape[0]
            )
            test_dis_data_set = self.strategy.experimental_distribute_dataset(test_data_set)
            loss = [distributed_train_step(data) for data in dis_data_set][0]
            # print("Successfully obtained gradients!")
            test_loss = [distributed_test_step(data) for data in test_dis_data_set][0]
            print_loss_table(i, loss, test_loss)
            # print('Epoch: {}, Loss: {}, Value Loss:{}, Measurable Loss:{}, Running Loss:{}, Terminal Loss{}, Test value loss {}, Test measurable loss{}.'
            #       .format(i + 1, loss[0], loss[1], loss[2], loss[3], loss[4], test_loss[1], test_loss[2]))

            loss_values.append(loss)  # The Neural Networks only be trained once per epoch.
            test_loss_values.append(test_loss) 
        self.loss_values = loss_values
        self.test_loss_values = test_loss_values
        return loss_values


if __name__ == '__main__':
    mu_1, mu_2 = tf.Variable(0.75, dtype=tf.float32), tf.Variable(0.25, dtype=tf.float32)
    sigma_1, sigma_2 = tf.Variable(0.4, dtype=tf.float32), tf.Variable(0.3, dtype=tf.float32)
    # ---------- build initial distribution and sampling ----------
    DIM_1_GMM = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical([0.3, 0.7]),
        components_distribution=tfp.distributions.Normal(
            loc=[mu_1, mu_2],
            scale=[sigma_1, sigma_2])
    )
    nn_config = 2, [64, 32, 32], 1
    test = Deep_FBSDE_Filter(delta_t=0.001, T=4.0,
                      drift=None, diffusion=None, sig_dim=1, obv_dim=1, init_dis=DIM_1_GMM,
                      nn_config=nn_config, build_neural_networks=build_Zakai_Solver)
