import streamlit as st
from quiz.quiz_utils import get_quiz

def render_quiz(algo_name):
    st.subheader(f"🧠 Quiz: {algo_name}")
    quiz_data = get_quiz(algo_name)
    if not quiz_data:
        st.info("Quiz not available for this algorithm yet.")
        return

    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}

    # Render each question without any option selected by default
    for idx, q in enumerate(quiz_data):
        st.markdown(f"**Q{idx+1}: {q['question']}**")

        # Get currently selected option or None if not answered yet
        current_answer = st.session_state.user_answers.get(idx, None)

        selected = st.radio(
            "Choose one:",
            q['options'],
            key=f"{algo_name}_{idx}",
            index=None if current_answer is None else q['options'].index(current_answer)
        )

        st.session_state.user_answers[idx] = selected

    if st.button("Submit All Answers"):
        score = 0
        for idx, q in enumerate(quiz_data):
            user_ans = st.session_state.user_answers.get(idx)
            if user_ans == q['answer']:
                st.success(f"Q{idx+1}: ✅ Correct!")
                score += 1
            else:
                st.error(f"Q{idx+1}: ❌ Wrong. Correct answer: **{q['answer']}**")

        st.markdown(f"### 🎯 Final Score: {score} / {len(quiz_data)}")
