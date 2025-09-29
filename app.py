import streamlit as st
from tutor.voice_guide import speak_theory
from tutor.learning_intro import show_learning_type_selector
from tutor import ai_tutor
from quiz import quiz_main
from dashboard import dashboard_main

# ---- THEORY MODE Imports ----

# Supervised
from visuals.plot_linear_regression import render as render_lr
from visuals.plot_logistic_regression import render as render_logr
from visuals.plot_decision_tree import render as render_dt
from visuals.plot_random_forest import render as render_rf
from visuals.plot_knn import render as render_knn
from visuals.plot_naive_bayes import render as render_nb
from visuals.plot_svm import render as render_svm

# Unsupervised
from visuals.plot_kmeans import render as render_kmeans
from visuals.plot_pca import render as render_pca
from visuals.plot_hierarchical import render as render_hier
from visuals.plot_dbscan import render as render_dbscan
from visuals.plot_autoencoder import render as render_auto
from visuals.plot_apriori import render as render_apriori

# Reinforcement
from visuals.plot_q_learning import render as render_ql
from visuals.plot_sarsa import render as render_sarsa
from visuals.plot_dqn import render as render_dqn
from visuals.plot_reinforce import render as render_reinforce
from visuals.plot_actor_critic import render as render_ac
from visuals.plot_ddpg import render as render_ddpg
from visuals.plot_a3c import render as render_a3c

# ---- EXPERIMENT MODE Imports ----

# Supervised
from experiment_mode.linear_regression_experiment import render as render_exp_lr
from experiment_mode.logistic_regression_experiment import render as render_exp_logr
from experiment_mode.decision_tree_experiment import render as render_exp_dt
from experiment_mode.random_forest_experiment import render as render_exp_rf
from experiment_mode.knn_experiment import render as render_exp_knn
from experiment_mode.naive_bayes_experiment import render as render_exp_nb
from experiment_mode.svm_experiment import render as render_exp_svm

# Unsupervised
from experiment_mode.kmeans_experiment import render as render_exp_kmeans
from experiment_mode.pca_experiment import render as render_exp_pca
from experiment_mode.hierarchical_experiment import render as render_exp_hier
from experiment_mode.dbscan_experiment import render as render_exp_dbscan
from experiment_mode.autoencoder_experiment import render as render_exp_auto
from experiment_mode.apriori_experiment import render as render_exp_apriori

# Reinforcement
from experiment_mode.q_learning_experiment import render as render_exp_ql
from experiment_mode.sarsa_experiment import render as render_exp_sarsa
from experiment_mode.dqn_experiment import render as render_exp_dqn
from experiment_mode.reinforce_experiment import render as render_exp_reinforce
from experiment_mode.actor_critic_experiment import render as render_exp_ac
from experiment_mode.ddpg_experiment import render as render_exp_ddpg
from experiment_mode.a3c_experiment import render as render_exp_a3c


def main():
    st.set_page_config(layout="wide", page_title="AlgoVista 3D – Study. Simulate. Succeed.")
    st.title("🚀 AlgoVista 3D – Study. Simulate. Succeed.")
    st.markdown("### 🧠 A Visual & Interactive ML Companion")

    # -------- Sidebar Navigation -------- #
    st.sidebar.title("🔍 Navigation")
    section = st.sidebar.radio("📚 Go To", [
        "📘 Learn Algorithms",
        "🤖 AI Tutor",
        "🧠 Quiz Time",
        "📊 User Dashboard"
    ])

    # -------- Learn Algorithms -------- #
    if section == "📘 Learn Algorithms":
        learning_type = show_learning_type_selector()
        mode = st.sidebar.radio("🎛️ Choose Mode", ["📘 Theory", "🧪 Experiment"])

        # 🔗 Mapping algorithms to their functions
        algo_options = {
            "Supervised Learning": {
                "Linear Regression": (render_lr, render_exp_lr),
                "Logistic Regression": (render_logr, render_exp_logr),
                "Decision Tree": (render_dt, render_exp_dt),
                "Random Forest": (render_rf, render_exp_rf),
                "K-Nearest Neighbors (KNN)": (render_knn, render_exp_knn),
                "Naive Bayes": (render_nb, render_exp_nb),
                "Support Vector Machine (SVM)": (render_svm, render_exp_svm),
            },
            "Unsupervised Learning": {
                "K-Means Clustering": (render_kmeans, render_exp_kmeans),
                "Principal Component Analysis (PCA)": (render_pca, render_exp_pca),
                "Hierarchical Clustering": (render_hier, render_exp_hier),
                "DBSCAN": (render_dbscan, render_exp_dbscan),
                "Autoencoders": (render_auto, render_exp_auto),
                "Apriori Algorithm": (render_apriori, render_exp_apriori),
            },
            "Reinforcement Learning": {
                "Q-Learning": (render_ql, render_exp_ql),
                "SARSA": (render_sarsa, render_exp_sarsa),
                "Deep Q-Network (DQN)": (render_dqn, render_exp_dqn),
                "REINFORCE Algorithm": (render_reinforce, render_exp_reinforce),
                "Actor-Critic": (render_ac, render_exp_ac),
                "DDPG": (render_ddpg, render_exp_ddpg),
                "A3C": (render_a3c, render_exp_a3c),
            }
        }

        if learning_type:
            st.write(f"🔎 **You selected:** `{learning_type}`")
            algo_choice = st.sidebar.selectbox("📊 Select Algorithm", list(algo_options[learning_type].keys()))

            with st.sidebar.expander("🔊 Voice Assistant"):
                if st.button("🔈 Play Theory Voiceover"):
                    speak_theory(algo_choice)

            st.markdown(f"### 🔍 Algorithm: {algo_choice}")
            st.divider()

            theory_func, experiment_func = algo_options[learning_type][algo_choice]

            if mode == "📘 Theory":
                theory_func()
            elif mode == "🧪 Experiment":
                if experiment_func:
                    experiment_func()
                else:
                    st.warning("⚠️ Experiment mode is not yet available for this algorithm.")

    # -------- AI Tutor Section -------- #
    elif section == "🤖 AI Tutor":
        ai_tutor.render()

    # -------- Quiz Section -------- #
    elif section == "🧠 Quiz Time":
        quiz_main.render()

    # -------- Dashboard Section -------- #
    elif section == "📊 User Dashboard":
        dashboard_main.render()


if __name__ == "__main__":
    main()