# models/reinforcement/reinforce.py

import numpy as np
import tensorflow as tf
from keras import layers
import gym

class REINFORCEAgent:
    def __init__(self, n_actions, input_dims, alpha=0.001, gamma=0.99):
        self.gamma = gamma
        self.lr = alpha
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        self.policy = self.build_policy_network(input_dims, n_actions)

    def build_policy_network(self, input_dims, n_actions):
        model = tf.keras.Sequential([
            layers.Dense(24, activation='relu', input_shape=input_dims),
            layers.Dense(24, activation='relu'),
            layers.Dense(n_actions, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss='categorical_crossentropy')
        return model

    def choose_action(self, state):
        state = state[np.newaxis, :]
        probabilities = self.policy.predict(state, verbose=0)[0]
        action = np.random.choice(len(probabilities), p=probabilities)
        return action

    def store_transition(self, state, action, reward):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
        G = np.zeros_like(self.reward_memory, dtype=np.float32)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum

        G = (G - np.mean(G)) / (np.std(G) + 1e-8)

        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state, action) in enumerate(zip(G, self.state_memory, self.action_memory)):
                state = state[np.newaxis, :]
                probs = self.policy(state, training=True)
                action_prob = probs[0, action]
                log_prob = tf.math.log(action_prob + 1e-8)
                loss += -g * log_prob

        grads = tape.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

        # Reset memory after learning
        self.state_memory.clear()
        self.action_memory.clear()
        self.reward_memory.clear()

        return loss.numpy()

def train_reinforce(env_name="CartPole-v1", episodes=100):
    env = gym.make(env_name)
    agent = REINFORCEAgent(n_actions=env.action_space.n, input_dims=(env.observation_space.shape[0],))
    scores = []

    for episode in range(episodes):
        done = False
        score = 0
        state = env.reset()[0]

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(state, action, reward)
            state = next_state
            score += reward
            done = terminated or truncated

        agent.learn()
        scores.append(score)

    env.close()
    return scores
