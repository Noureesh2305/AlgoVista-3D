# dashboard/dashboard_main.py

import streamlit as st
import time

def render():
    st.subheader("📊 User Dashboard – Your ML Learning Journey")

    st.markdown("""
    Welcome to your personalized dashboard! Here's an overview of your activity.
    
    > ⚠️ *Currently running in Guest Mode – no login or storage enabled.*  
    > 📌 *Your progress will reset if you refresh. Persistent save (SQLite/Firebase) coming soon!*
    """)

    # Ensure stats exist in session_state
    if "stats" not in st.session_state:
        st.session_state.stats = {
            "algorithms_visited": 0,
            "quizzes_attempted": 0,
            "voiceovers_played": 0,
            "ai_tutor_questions": 0,
            "experiments_tried": 0
        }

    stats = st.session_state.stats

    # Section 1: Key Stats (dynamic)
    st.metric("🧠 Algorithms Visited", stats["algorithms_visited"])
    st.metric("📝 Quizzes Attempted", stats["quizzes_attempted"])
    st.metric("🎧 Voiceovers Played", stats["voiceovers_played"])
    st.metric("🤖 AI Tutor Questions", stats["ai_tutor_questions"])
    st.metric("🧪 Experiments Tried", stats["experiments_tried"])

    # Section 2: Timeline Chart (simulated until real data logging is added)
    st.markdown("---")
    with st.expander("📈 Activity Timeline"):
        # Example: Using algorithms visited count to simulate "hours spent"
        hours_data = [i * 0.8 for i in range(1, stats["algorithms_visited"] + 1)]
        if hours_data:
            st.line_chart({"Hours": hours_data})
        else:
            st.info("No activity recorded yet to show timeline.")

    # Section 3: Motivation
    st.success("🎯 Goal: Complete all theory + quiz + experiment modules to master ML!")
    st.info("🔐 Login & Progress Save – Coming Soon 🚀")
