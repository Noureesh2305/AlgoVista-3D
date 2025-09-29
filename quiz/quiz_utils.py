import random

QUIZ_QUESTIONS = {
    "Linear Regression": [
        {
            "question": "What is the main assumption of linear regression?",
            "options": ["Non-linear relationship", "Linearity between variables", "High multicollinearity", "Categorical inputs"],
            "answer": "Linearity between variables"
        },
        {
            "question": "Which metric is commonly used to evaluate linear regression?",
            "options": ["Accuracy", "RMSE", "Recall", "Entropy"],
            "answer": "RMSE"
        },
        {
            "question": "What does the coefficient represent in linear regression?",
            "options": ["Slope", "Intercept", "Error", "Mean"],
            "answer": "Slope"
        },
        {
            "question": "Which technique helps prevent overfitting in linear regression?",
            "options": ["Regularization", "Normalization", "Clustering", "PCA"],
            "answer": "Regularization"
        },
        {
            "question": "Linear regression is used for which type of problem?",
            "options": ["Classification", "Regression", "Clustering", "Dimensionality reduction"],
            "answer": "Regression"
        },
        {
            "question": "Which method is used to estimate the parameters in linear regression?",
            "options": ["Gradient Descent", "K-Means", "Backpropagation", "Expectation-Maximization"],
            "answer": "Gradient Descent"
        },
        {
            "question": "What is multicollinearity?",
            "options": ["Correlation between predictors", "Overfitting", "Missing values", "Outliers"],
            "answer": "Correlation between predictors"
        },
    ],
    "K-Means Clustering": [
        {
            "question": "What does K mean in K-Means?",
            "options": ["Clusters", "Features", "Samples", "Dimensions"],
            "answer": "Clusters"
        },
        {
            "question": "Which distance metric is commonly used in K-Means?",
            "options": ["Euclidean", "Manhattan", "Hamming", "Jaccard"],
            "answer": "Euclidean"
        },
        {
            "question": "K-Means is an example of which type of learning?",
            "options": ["Supervised", "Unsupervised", "Reinforcement", "Semi-supervised"],
            "answer": "Unsupervised"
        },
        {
            "question": "What is updated during each iteration in K-Means?",
            "options": ["Centroids", "Labels", "Weights", "Features"],
            "answer": "Centroids"
        },
        {
            "question": "K-Means clustering is sensitive to:",
            "options": ["Initialization", "Data scaling", "Outliers", "All of the above"],
            "answer": "All of the above"
        },
        {
            "question": "What is the objective function minimized by K-Means?",
            "options": ["Sum of squared distances to centroids", "Entropy", "Mutual information", "Likelihood"],
            "answer": "Sum of squared distances to centroids"
        },
        {
            "question": "Which step assigns points to the nearest cluster centroid?",
            "options": ["Expectation step", "Maximization step", "Backpropagation", "Sampling"],
            "answer": "Expectation step"
        },
    ],
    "Support Vector Machine (SVM)": [
        {
            "question": "What is the goal of SVM?",
            "options": ["Maximize margin", "Minimize loss", "Reduce dimensionality", "Increase accuracy"],
            "answer": "Maximize margin"
        },
        {
            "question": "SVM is mainly used for:",
            "options": ["Classification", "Regression", "Clustering", "Data generation"],
            "answer": "Classification"
        },
        {
            "question": "What kernel would you use for non-linearly separable data?",
            "options": ["Linear", "Polynomial", "RBF", "Both Polynomial and RBF"],
            "answer": "Both Polynomial and RBF"
        },
        {
            "question": "What is a support vector in SVM?",
            "options": ["A data point closest to the hyperplane", "A random point", "The centroid of data", "An outlier"],
            "answer": "A data point closest to the hyperplane"
        },
        {
            "question": "SVM tries to find a hyperplane that:",
            "options": ["Separates classes with maximum margin", "Minimizes number of support vectors", "Maximizes number of misclassifications", "Fits data perfectly"],
            "answer": "Separates classes with maximum margin"
        },
    ],
    "Principal Component Analysis (PCA)": [
        {
            "question": "What is the main purpose of PCA?",
            "options": ["Reduce dimensionality", "Increase dimensionality", "Cluster data", "Classify data"],
            "answer": "Reduce dimensionality"
        },
        {
            "question": "PCA creates new features called:",
            "options": ["Principal Components", "Clusters", "Centroids", "Weights"],
            "answer": "Principal Components"
        },
        {
            "question": "PCA tries to maximize:",
            "options": ["Variance", "Distance", "Entropy", "Accuracy"],
            "answer": "Variance"
        },
        {
            "question": "PCA is an example of:",
            "options": ["Supervised learning", "Unsupervised learning", "Reinforcement learning", "Semi-supervised learning"],
            "answer": "Unsupervised learning"
        },
    ],
    "Decision Tree": [
        {
            "question": "What criterion is commonly used to split nodes in a decision tree?",
            "options": ["Gini Impurity", "Entropy", "Mean Squared Error", "Both Gini Impurity and Entropy"],
            "answer": "Both Gini Impurity and Entropy"
        },
        {
            "question": "Decision trees are prone to:",
            "options": ["Overfitting", "Underfitting", "High bias", "None of the above"],
            "answer": "Overfitting"
        },
        {
            "question": "What is pruning in decision trees?",
            "options": ["Reducing tree size", "Growing tree fully", "Adding more branches", "Feature selection"],
            "answer": "Reducing tree size"
        },
        {
            "question": "Leaves in decision trees represent:",
            "options": ["Predicted output", "Feature split", "Data clusters", "Errors"],
            "answer": "Predicted output"
        },
    ],
    "Random Forest": [
        {
            "question": "Random Forest uses which technique?",
            "options": ["Bagging", "Boosting", "Stacking", "None of the above"],
            "answer": "Bagging"
        },
        {
            "question": "Random Forest helps to reduce:",
            "options": ["Overfitting", "Underfitting", "Noise", "Bias"],
            "answer": "Overfitting"
        },
        {
            "question": "Each tree in Random Forest is trained on:",
            "options": ["Bootstrap samples", "All data", "Random features only", "Validation set"],
            "answer": "Bootstrap samples"
        },
        {
            "question": "Random Forest combines predictions using:",
            "options": ["Majority voting", "Averaging", "Both depending on task", "Random selection"],
            "answer": "Both depending on task"
        },
    ],
    # Add more algorithms and questions as needed
}

def get_quiz(algorithm, num_questions=5):
    questions = QUIZ_QUESTIONS.get(algorithm, [])
    if len(questions) > num_questions:
        questions = random.sample(questions, num_questions)
    else:
        random.shuffle(questions)
    return questions
