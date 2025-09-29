import streamlit as st
from quiz.quiz_utils import get_quiz

def render():
    st.subheader("🧠 Quiz Time – Test Your ML Knowledge")
    st.markdown("Answer the following questions. ⭐ 1 point for each correct answer.")

    algo_name = "Linear Regression"  # Change dynamically if needed

    # Initialize quiz questions only once per session
    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = get_quiz(algo_name)

    # Initialize user answers dict if not exists
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}

    questions = st.session_state.quiz_questions

    # Show questions with saved selected answers (or None)
    for i, q in enumerate(questions):
        st.markdown(f"**Q{i+1}: {q['question']}**")

        # Get previously selected answer or None
        default_index = None
        if f"q_{i}" in st.session_state.user_answers:
            try:
                default_index = q['options'].index(st.session_state.user_answers[f"q_{i}"])
            except ValueError:
                default_index = None

        selected = st.radio(
            "Select your answer:",
            q["options"],
            index=default_index,
            key=f"q_{i}"
        )
        # Update user_answers on selection change
        st.session_state.user_answers[f"q_{i}"] = selected

        st.markdown("---")

    # Submit button inside render function, with access to 'questions'
    if st.button("Submit Answers"):
        score = 0
        for i, q in enumerate(questions):
            user_choice = st.session_state.user_answers.get(f"q_{i}")
            if user_choice is None:
                st.warning(f"Q{i+1}: No answer selected.")
            elif user_choice == q["answer"]:
                st.success(f"Q{i+1}: Correct! ✅")
                score += 1
            else:
                st.error(f"Q{i+1}: Incorrect ❌. Correct answer: **{q['answer']}**")

        st.markdown(f"### 🎯 Your Total Score: {score} out of {len(questions)}")
        st.balloons()

        # Increment the quiz attempt counter in session_state
        if "quizzes_attempted" not in st.session_state:
            st.session_state.quizzes_attempted = 0
        st.session_state.quizzes_attempted += 1

        # Uncomment below if you want to reset quiz after submit
        # del st.session_state.quiz_questions
        # del st.session_state.user_answers
