"""
Streamlit app for Student Performance Predictor
- Shows model status
- Allows training (on synthetic or upload)
- Predicts, logs, chats, and generates PDF reports
"""

import streamlit as st
import pandas as pd
from train_model import train_and_save, generate_synthetic_data
from utils import (
    load_artifacts, MODEL_PATH, ENCODER_PATH,
    predict_student as util_predict_student,
    log_performance, read_logs, generate_report, chatbot_answer
)
import joblib
import os

st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("🎓 Student Performance Predictor & Academic Assistant")

# Reload artifacts helper
def reload_artifacts():
    global MODEL, LABEL_ENCODER
    try:
        MODEL = joblib.load(MODEL_PATH)
        LABEL_ENCODER = joblib.load(ENCODER_PATH)
        return True
    except Exception:
        MODEL = None
        LABEL_ENCODER = None
        return False

# Initial artifact state
MODEL_EXISTS = os.path.exists("saved_model.pkl") and os.path.exists("label_encoder.pkl")
if MODEL_EXISTS:
    MODEL, LABEL_ENCODER = load_artifacts()
else:
    MODEL = LABEL_ENCODER = None

# Sidebar controls
st.sidebar.header("Controls")
if st.sidebar.button("Train model (synthetic)"):
    with st.spinner("Training model (this may take 10-30s)..."):
        model, le, summary = train_and_save()
        # reload artifacts
        MODEL, LABEL_ENCODER = load_artifacts()
    st.sidebar.success("Model trained & saved.")
    st.sidebar.text_area("Train summary", value=summary, height=200)

st.sidebar.markdown("---")
st.sidebar.write("If you have a real CSV with the required columns (StudyHours, Attendance, AssignmentsCompleted, PerformanceLevel), upload and click 'Train on uploaded CSV'.")

uploaded = st.sidebar.file_uploader("Upload CSV to train (optional)", type=["csv"])
if uploaded is not None:
    df_upload = pd.read_csv(uploaded)
    if st.sidebar.button("Train on uploaded CSV"):
        try:
            model, le, summary = train_and_save(df=df_upload)
            MODEL, LABEL_ENCODER = load_artifacts()
            st.sidebar.success("Trained on uploaded CSV.")
            st.sidebar.text_area("Train summary", value=summary, height=200)
        except Exception as e:
            st.sidebar.error(f"Training failed: {e}")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Predict & Log", "Chatbot", "Logs", "Generate Report"])

with tab1:
    st.header("📊 Predict Student Performance")
    col1, col2 = st.columns(2)
    with col1:
        student_id = st.text_input("Student ID", value="S001")
        week = st.number_input("Week", min_value=1, max_value=52, value=1)
        study_hours = st.slider("Study Hours per day (approx)", 0, 24, 3)
    with col2:
        attendance = st.slider("Attendance (%)", 0, 100, 75)
        assignments = st.number_input("Assignments Completed (count)", min_value=0, max_value=50, value=5)
        st.write("")

    if st.button("Predict & Log"):
        # Safe predict with useful messages
        try:
            # Try to predict using utils.predict_student
            pred_label = util_predict_student(study_hours, attendance, assignments)
            st.success(f"Predicted Performance: **{pred_label}**")
            # Log
            log_performance(student_id, week, study_hours, attendance, assignments, pred_label)
            st.info("Logged successfully (updates previous entry for same StudentID+Week).")
        except FileNotFoundError:
            st.error("Model not trained. Use the sidebar to train the model first.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

with tab2:
    st.header("💬 Academic Assistant Chatbot")
    q = st.text_area("Ask any study-related question (e.g., 'Give me a study plan for 2 months')")
    if st.button("Get Advice"):
        if not q.strip():
            st.warning("Please write a question.")
        else:
            with st.spinner("Getting reply..."):
                ans = chatbot_answer(q)
            st.markdown("**Assistant:**")
            st.write(ans)

with tab3:
    st.header("📑 Performance Logs")
    df_logs = read_logs()
    if df_logs.empty:
        st.info("No logs yet. Make a prediction and log a student first.")
    else:
        st.dataframe(df_logs.sort_values(["StudentID", "Week"], ascending=[True, True]))

        if st.button("Clear all logs (danger)"):
            if st.confirm("Are you sure you want to delete all logs?"):
                os.remove("performance_log.csv")
                st.experimental_rerun()

with tab4:
    st.header("📄 Generate Student PDF Report")
    student_for_report = st.text_input("Student ID for report", value="S001")
    if st.button("Generate PDF Report"):
        try:
            path = generate_report(student_for_report.strip())
            with open(path, "rb") as f:
                st.download_button("Download PDF", f, file_name=os.path.basename(path))
            st.success("Report generated.")
        except Exception as e:
            st.error(f"Could not generate report: {e}")

st.markdown("---")
st.caption("✅ Project is prepared for placement: model training, logging, report export, and a chatbot (OpenAI if key provided). Keep your API keys in environment variables (OPENAI_API_KEY).")
