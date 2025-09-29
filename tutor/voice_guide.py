# tutor/voice_guide.py

import pyttsx3

# Predefined theory explanations for each algorithm
THEORY = {
    # Supervised
    "Linear Regression": "Linear Regression is a supervised learning algorithm that models the relationship between a dependent variable and one or more independent variables using a straight line.",
    "Logistic Regression": "Logistic Regression is used for binary classification problems. It estimates the probability of a binary response using the logistic function.",
    "K-Nearest Neighbors (KNN)": "KNN is a non-parametric algorithm that classifies a data point based on how its neighbors are classified. It uses distance metrics like Euclidean.",
    "Decision Tree": "Decision Tree builds a tree-like structure where internal nodes represent decisions based on features, and leaves represent outcomes.",
    "Random Forest": "Random Forest is an ensemble learning method using multiple decision trees and majority voting to improve accuracy and control overfitting.",
    "Naive Bayes": "Naive Bayes is a probabilistic classifier based on Bayes’ Theorem with the assumption of feature independence.",
    "Support Vector Machine (SVM)": "SVM is a supervised learning model that tries to find the optimal hyperplane that separates data into different classes.",

    # Unsupervised
    "K-Means Clustering": "K-Means is an unsupervised clustering algorithm that partitions data into K clusters based on feature similarity.",
    "Principal Component Analysis (PCA)": "PCA is a dimensionality reduction technique that transforms data into new components with maximum variance.",
    "Hierarchical Clustering": "Hierarchical Clustering builds a tree of clusters, merging or splitting groups step-by-step to form a hierarchy.",
    "DBSCAN": "DBSCAN is a density-based clustering algorithm that groups closely packed points together and marks outliers in low-density regions.",
    "Autoencoders": "Autoencoders are neural networks that learn to compress and reconstruct input data. They're used for anomaly detection and dimensionality reduction.",
    "Apriori Algorithm": "Apriori is used in market basket analysis to find frequent itemsets and generate association rules from transactional data.",

    # Reinforcement Learning
    "Q-Learning": "Q-Learning is a model-free reinforcement learning algorithm where the agent learns optimal actions using Q-values updated through exploration.",
    "SARSA": "SARSA is a reinforcement learning algorithm that updates values based on the action the agent actually takes, making it more cautious than Q-learning.",
    "Deep Q Network (DQN)": "DQN uses deep neural networks to approximate Q-values, enabling agents to learn complex policies in environments like video games.",
    "REINFORCE": "REINFORCE is a policy gradient method in reinforcement learning that updates policy weights based on rewards received at the end of each episode.",
    "Actor-Critic": "Actor-Critic combines two models: the actor, which chooses actions, and the critic, which evaluates how good the action was.",
    "DDPG": "DDPG is a reinforcement learning algorithm used in continuous action spaces, combining actor-critic methods with deep learning for smooth control.",
    "A3C": "A3C, or Asynchronous Advantage Actor Critic, runs multiple agents in parallel to learn faster and more robustly in complex environments.",
}

def speak_theory(algorithm_name):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)  # Adjust speed
    text = THEORY.get(algorithm_name, "Sorry, theory not available for this algorithm.")
    engine.say(text)
    engine.runAndWait()
