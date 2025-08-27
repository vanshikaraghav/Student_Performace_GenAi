
import streamlit as st
import pandas as pd
import os
from train_model import load_ucistudent_data, build_and_train, load_model, predict_g3_from_row
from utils import log_prediction, read_logs, generate_pdf_report, generate_intervention_plan, ask_hf_system

st.set_page_config(page_title="Student Performance GenAI", layout="wide")
st.title("🎓 Student Performance GenAI (ML + HF Chatbot)")

# -------------------------
# Sidebar - Model training
# -------------------------
st.sidebar.header("Model & Data Controls")
if st.sidebar.button("Train model on UCI dataset"):
    with st.spinner("Fetching dataset and training..."):
        df = load_ucistudent_data()
        model_pipe, report = build_and_train(df)
        st.sidebar.success("Model trained and saved.")
        st.sidebar.text_area("Train summary", value=report, height=180)

uploaded = st.sidebar.file_uploader("Upload CSV to train (must include 'G3')", type=["csv"])
if uploaded is not None and st.sidebar.button("Train on uploaded CSV"):
    try:
        df_up = pd.read_csv(uploaded)
        model_pipe, report = build_and_train(df_up)
        st.sidebar.success("Trained on uploaded CSV.")
        st.sidebar.text_area("Train summary", value=report, height=180)
    except Exception as e:
        st.sidebar.error(f"Training failed: {e}")

# Load model
pipe = load_model("saved_model.pkl") if os.path.exists("saved_model.pkl") else None

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Predict & Log", "Chatbot", "Logs", "Reports"])

# -------------------------
# Tab 1: Predict & Log
# -------------------------
with tab1:
    st.header("📊 Predict & Log Student Performance (G3)")
    col1, col2 = st.columns(2)

    # --- Manual Input ---
    with col1:
        st.subheader("Manual Input")
        student_id = st.text_input("Student ID", value="S001")
        week = st.number_input("Week", min_value=1, max_value=52, value=1)

        sex = st.selectbox("Sex", ["F","M"])
        age = st.number_input("Age", 15, 22, 16)
        school = st.selectbox("School", ["GP","MS"])
        address = st.selectbox("Address", ["U","R"])
        famsize = st.selectbox("Family size", ["LE3","GT3"])
        Pstatus = st.selectbox("Parent status", ["T","A"])
        Medu = st.selectbox("Mother education", [0,1,2,3,4])
        Fedu = st.selectbox("Father education", [0,1,2,3,4])
        Mjob = st.selectbox("Mother job", ["teacher","health","services","at_home","other"])
        Fjob = st.selectbox("Father job", ["teacher","health","services","at_home","other"])
        reason = st.selectbox("Reason to choose school", ["home","reputation","course","other"])
        guardian = st.selectbox("Guardian", ["mother","father","other"])
        traveltime = st.selectbox("Travel time", [1,2,3,4])
        studytime = st.selectbox("Study time", [1,2,3,4])
        failures = st.number_input("Past failures", 0,3,0)
        schoolsup = st.selectbox("School support", ["yes","no"])
        famsup = st.selectbox("Family support", ["yes","no"])
        paid = st.selectbox("Extra paid classes", ["yes","no"])
        activities = st.selectbox("Extra-curricular activities", ["yes","no"])
        nursery = st.selectbox("Attended nursery", ["yes","no"])
        higher = st.selectbox("Wants higher education", ["yes","no"])
        internet = st.selectbox("Internet access", ["yes","no"])
        romantic = st.selectbox("Romantic relationship", ["yes","no"])
        famrel = st.number_input("Family relationship quality",1,5,4)
        freetime = st.number_input("Free time",1,5,3)
        goout = st.selectbox("Going out with friends", [1,2,3,4])
        Dalc = st.selectbox("Workday alcohol consumption", [1,2,3,4,5])
        Walc = st.selectbox("Weekend alcohol consumption", [1,2,3,4,5])
        health = st.selectbox("Current health status", [1,2,3,4,5])
        absences = st.number_input("Absences",0,93,4)

    # --- CSV upload ---
    with col2:
        st.subheader("CSV Upload (single row)")
        uploaded_row = st.file_uploader("Upload CSV with same columns", type=["csv"], key="row")
        df_row = pd.read_csv(uploaded_row) if uploaded_row else None
        if df_row is not None:
            st.dataframe(df_row.head(1))

    if st.button("Predict & Log"):
        if df_row is not None:
            row = df_row.iloc[0].to_dict()
        else:
            row = {
                "sex": sex, "age": age, "school": school, "address": address,
                "famsize": famsize, "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu,
                "Mjob": Mjob, "Fjob": Fjob, "reason": reason, "guardian": guardian,
                "traveltime": traveltime, "studytime": studytime, "failures": failures,
                "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
                "activities": activities, "nursery": nursery, "higher": higher,
                "internet": internet, "romantic": romantic,"famrel": famrel, "freetime": freetime, "goout": goout,
                "Dalc": Dalc, "Walc": Walc, "health": health, "absences": absences
            }

        if pipe is None:
            st.error("Model not available. Train the model first.")
        else:
            try:
                s = pd.Series(row)
                predicted = predict_g3_from_row(pipe, s)
                st.success(f"Predicted G3: {predicted:.2f} / 20")
                log_prediction(student_id, week, row, predicted)
                st.info("Prediction logged.")

                if st.checkbox("Generate quick intervention plan"):
                    plan = generate_intervention_plan(row, predicted)
                    st.subheader("Suggested Plan")
                    st.write(plan)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# -------------------------
# Tab 2: Chatbot
# -------------------------
with tab2:
    st.header("💬 AI Chatbot")
    question = st.text_area("Ask a study-help question:", height=120)
    if st.button("Ask"):
        if question.strip():
            answer = ask_hf_system(question)
            st.markdown("**Assistant:**")
            st.write(answer)
        else:
            st.warning("Type a question first.")

# -------------------------
# Tab 3: Logs
# -------------------------
with tab3:
    st.header("📑 Logs")
    logs = read_logs()
    if logs.empty:
        st.info("No logs yet.")
    else:
        st.dataframe(logs.sort_values(["StudentID","Week"]))
        if st.button("Clear all logs"):
            try:
                os.remove("performance_log.csv")
                st.success("Logs cleared.")
            except Exception as e:
                st.error(f"Failed to clear logs: {e}")

# -------------------------
# Tab 4: Reports
# -------------------------
with tab4:
    st.header("📄 Reports")
    sid = st.text_input("Student ID for report", value="S001")
    if st.button("Generate PDF Report"):
        try:
            path = generate_pdf_report(sid.strip())
            with open(path, "rb") as f:
                st.download_button("Download PDF", f, file_name=os.path.basename(path))
            st.success("Report generated.")
        except Exception as e:
            st.error(f"Could not generate report: {e}")

st.markdown("---")
st.caption("Project: Student Performance (UCI dataset) + HF Chatbot")

