import tensorflow as tf
import numpy as np
import gym
from tensorflow.keras.layers import Dense

class PPOActorCritic:
    def __init__(self, state_dim, action_dim, action_bound, lr_actor, lr_critic, gamma, epsilon):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.epsilon = epsilon

        # self.sess = tf.Session()

        # self._build_networks()
        # self.sess.run(tf.global_variables_initializer())
        
        # Actor and Critic models
        self.actor_model = self._build_actor()
        self.critic_model = self._build_critic()

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(self.lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.lr_critic)
    
    def _build_actor(self):
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        mu = Dense(self.action_dim, activation='tanh')(x)
        sigma = Dense(self.action_dim, activation='softplus')(x)
        model = tf.keras.Model(inputs, [mu * self.action_bound, sigma])
        return model

    def _build_critic(self):
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        value = Dense(1)(x)
        model = tf.keras.Model(inputs, value)
        return model

    def get_action_and_log_prob(self, state):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        mu, sigma = self.actor_model(state)
        dist = tf.random.normal(tf.shape(mu), mean=mu, stddev=sigma)
        sampled_action = tf.clip_by_value(dist, -self.action_bound, self.action_bound)

        # Log probability
        log_prob = -0.5 * (((sampled_action - mu) / sigma) ** 2 + 2 * tf.math.log(sigma) + tf.math.log(2 * tf.constant(3.1415)))
        log_prob = tf.reduce_sum(log_prob, axis=-1, keepdims=True)

        return sampled_action.numpy(), log_prob.numpy()

    def get_value(self, state):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        return self.critic_model(state).numpy()

    def update(self, states, actions, advantages, target_values, old_log_probs):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        target_values = tf.convert_to_tensor(target_values, dtype=tf.float32)
        old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)

        # Update actor
        with tf.GradientTape() as tape:
            mu, sigma = self.actor_model(states)
            dist = tf.random.normal(tf.shape(mu), mean=mu, stddev=sigma)
            log_probs = -0.5 * (((dist - mu) / sigma) ** 2 + 2 * tf.math.log(sigma) + tf.math.log(2 * tf.constant(3.1415)))
            log_probs = tf.reduce_sum(log_probs, axis=-1, keepdims=True)

            ratio = tf.exp(log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon)
            surrogate_loss = tf.minimum(ratio * advantages, clipped_ratio * advantages)
            actor_loss = -tf.reduce_mean(surrogate_loss)

        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

        # Update critic
        with tf.GradientTape() as tape:
            value_preds = self.critic_model(states)
            critic_loss = tf.reduce_mean(tf.square(target_values - value_preds))

        critic_grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))

    # def _build_networks(self):
    #     self.state = tf.placeholder(tf.float32, [None, self.state_dim], 'state')
    #     self.action = tf.placeholder(tf.float32, [None, self.action_dim], 'action')
    #     self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
    #     self.target_value = tf.placeholder(tf.float32, [None, 1], 'target_value')
    #     self.old_log_prob = tf.placeholder(tf.float32, [None, 1], 'old_log_prob')

    #     # Actor network
    #     with tf.variable_scope('actor'):
    #         actor_net = tf.layers.dense(self.state, 64, activation=tf.nn.relu)
    #         actor_net = tf.layers.dense(actor_net, 64, activation=tf.nn.relu)
    #         mu = tf.layers.dense(actor_net, self.action_dim, activation=tf.nn.tanh) * self.action_bound
    #         sigma = tf.layers.dense(actor_net, self.action_dim, activation=tf.nn.softplus)

    #         self.dist = tf.distributions.Normal(loc=mu, scale=sigma)
    #         self.sampled_action = tf.clip_by_value(self.dist.sample(), -self.action_bound, self.action_bound)
    #         self.log_prob = self.dist.log_prob(self.action)

    #     # Critic network
    #     with tf.variable_scope('critic'):
    #         critic_net = tf.layers.dense(self.state, 64, activation=tf.nn.relu)
    #         critic_net = tf.layers.dense(critic_net, 64, activation=tf.nn.relu)
    #         self.value = tf.layers.dense(critic_net, 1)

    #     # Actor loss
    #     ratio = tf.exp(self.log_prob - self.old_log_prob)
    #     clipped_ratio = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon)
    #     surrogate_loss = tf.minimum(ratio * self.advantage, clipped_ratio * self.advantage)
    #     self.actor_loss = -tf.reduce_mean(surrogate_loss)

    #     # Critic loss
    #     self.critic_loss = tf.reduce_mean(tf.square(self.target_value - self.value))

    #     # Optimizers
    #     self.train_actor = tf.train.AdamOptimizer(self.lr_actor).minimize(self.actor_loss)
    #     self.train_critic = tf.train.AdamOptimizer(self.lr_critic).minimize(self.critic_loss)

    # def update(self, states, actions, advantages, target_values, old_log_probs):
    #     self.sess.run([self.train_actor, self.train_critic], feed_dict={
    #         self.state: states,
    #         self.action: actions,
    #         self.advantage: advantages,
    #         self.target_value: target_values,
    #         self.old_log_prob: old_log_probs
    #     })

    # def get_action_and_log_prob(self, state):
    #     # state = state[np.newaxis, :]
    #     # action, log_prob = self.sess.run([self.sampled_action, self.log_prob], feed_dict={self.state: state})
    #     # return action[0], log_prob[0]
    #     # Compute sampled_action first
    #     sampled_action = self.sess.run(self.sampled_action, feed_dict={self.state: state})

    #     # Compute log_prob using the sampled action
    #     log_prob = self.sess.run(self.log_prob, feed_dict={self.state: state, self.action: sampled_action})

    #     return sampled_action[0], log_prob[0]

    # def get_value(self, state):
    #     state = state[np.newaxis, :]
    #     return self.sess.run(self.value, feed_dict={self.state: state})[0]

#########################  Memory  #########################
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []

    def store(self, state, action, reward, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()

    def compute_advantages_and_targets(self, gamma, lam):
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        advantages = np.zeros_like(rewards)
        target_values = np.zeros_like(rewards)

        # Compute advantages and targets using Generalized Advantage Estimation (GAE)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * (values[t + 1] if t + 1 < len(rewards) else 0) - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
            target_values[t] = rewards[t] + gamma * (values[t + 1] if t + 1 < len(rewards) else 0)

        return advantages, target_values

#### This implementaion is for Gym environment. Need to change it to Energy Harvesting Wireless Communication environment

#########################  Training Loop  #########################
# env = gym.make('Pendulum-v1')
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
# action_bound = env.action_space.high[0]

# ppo = PPOActorCritic(state_dim, action_dim, action_bound, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, epsilon=0.2)
# memory = Memory()

# for episode in range(1000):
#     state = env.reset()
#     total_reward = 0

#     for step in range(200):
#         action, log_prob = ppo.get_action_and_log_prob(state)
#         value = ppo.get_value(state)

#         next_state, reward, done, _ = env.step(action)
#         memory.store(state, action, reward, value, log_prob)

#         state = next_state
#         total_reward += reward

#         if done or step == 199:
#             advantages, target_values = memory.compute_advantages_and_targets(ppo.gamma, lam=0.95)
#             ppo.update(np.array(memory.states), np.array(memory.actions), advantages[:, None], target_values[:, None], np.array(memory.log_probs)[:, None])
#             memory.clear()
#             break

#     print(f"Episode {episode + 1}: Total Reward = {total_reward}")
