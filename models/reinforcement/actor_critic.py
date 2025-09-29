# models/reinforcement/actor_critic.py

import gym
import numpy as np
import tensorflow as tf
from keras import layers

def train_actor_critic(episodes=300):
    """
    Actor-Critic method combines:
    - Actor: selects action based on policy
    - Critic: evaluates action using value function

    This function trains on CartPole for demo purposes.
    """
    env = gym.make("CartPole-v1")
    num_actions = env.action_space.n
    num_states = env.observation_space.shape[0]

    gamma = 0.99  # discount factor

    # Actor: Policy network
    actor_model = tf.keras.Sequential([
        layers.Input(shape=(num_states,)),
        layers.Dense(24, activation='relu'),
        layers.Dense(num_actions, activation='softmax')
    ])

    # Critic: Value network
    critic_model = tf.keras.Sequential([
        layers.Input(shape=(num_states,)),
        layers.Dense(24, activation='relu'),
        layers.Dense(1)
    ])

    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

    for ep in range(episodes):
        state = env.reset()[0]
        state = np.reshape(state, [1, num_states])
        done = False
        total_reward = 0

        while not done:
            probs = actor_model(state)
            action = np.random.choice(num_actions, p=np.squeeze(probs))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, num_states])

            # Compute advantage
            value = critic_model(state)
            next_value = critic_model(next_state)
            td_target = reward + gamma * next_value * (1 - int(done))
            advantage = td_target - value

            # Update Actor
            with tf.GradientTape() as tape1:
                probs = actor_model(state)
                action_prob = tf.math.log(probs[0, action])
                loss_actor = -action_prob * advantage
            grads1 = tape1.gradient(loss_actor, actor_model.trainable_variables)
            actor_optimizer.apply_gradients(zip(grads1, actor_model.trainable_variables))

            # Update Critic
            with tf.GradientTape() as tape2:
                v = critic_model(state)
                loss_critic = tf.square(td_target - v)
            grads2 = tape2.gradient(loss_critic, critic_model.trainable_variables)
            critic_optimizer.apply_gradients(zip(grads2, critic_model.trainable_variables))

            state = next_state
            total_reward += reward

    return actor_model  # For visualization, this return can be optional
