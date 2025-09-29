# models/reinforcement/ddpg.py

import gym
import numpy as np
import tensorflow as tf
from keras import layers

def get_actor(state_dim, action_dim):
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(action_dim, activation="tanh")(x)
    return tf.keras.Model(inputs, outputs)

def get_critic(state_dim, action_dim):
    state_input = layers.Input(shape=(state_dim,))
    action_input = layers.Input(shape=(action_dim,))
    concat = layers.Concatenate()([state_input, action_input])
    x = layers.Dense(256, activation="relu")(concat)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    return tf.keras.Model([state_input, action_input], outputs)

def train_ddpg(episodes=10):
    env = gym.make("Pendulum-v1", render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = get_actor(state_dim, action_dim)
    critic = get_critic(state_dim, action_dim)

    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

    for ep in range(episodes):
        state = env.reset()[0]
        done = False
        while not done:
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            action = actor(state_tensor)[0].numpy()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # Skipping replay buffer and training for simplicity
            state = next_state

    return actor
