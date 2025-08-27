"""
utils.py

Utilities: safe model/encoder loading, prediction (DataFrame-based),
logging (with update on duplicate StudentID+Week), PDF report generation,
and a simple Chatbot wrapper with graceful fallback if OpenAI key missing.
"""

import os
import pandas as pd
import joblib
from datetime import datetime
from fpdf import FPDF
import matplotlib.pyplot as plt
from io import BytesIO
import openai

MODEL_PATH = "saved_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
LOG_PATH = "performance_log.csv"

# -------------------------
# Safe load model & encoder
# -------------------------
def load_artifacts():
    model = None
    le = None
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None

    try:
        le = joblib.load(ENCODER_PATH)
    except Exception:
        le = None

    return model, le

MODEL, LABEL_ENCODER = load_artifacts()

# -------------------------
# Prediction (DataFrame)
# -------------------------
def predict_student(study_hours, attendance, assignments):
    """
    Returns a string label (e.g., 'High') or raises a helpful error.
    """
    global MODEL, LABEL_ENCODER
    if MODEL is None or LABEL_ENCODER is None:
        raise FileNotFoundError("Model or label encoder not found. Please run training.")

    features = pd.DataFrame([{
        "StudyHours": int(study_hours),
        "Attendance": int(attendance),
        "AssignmentsCompleted": int(assignments)
    }])
    pred_encoded = MODEL.predict(features)[0]
    label = LABEL_ENCODER.inverse_transform([int(pred_encoded)])[0]
    return label

# -------------------------
# Logging (update duplicates)
# -------------------------
def log_performance(student_id, week, study_hours, attendance, assignments, prediction):
    """
    Save log with full inputs. If same StudentID+Week exists, update that row.
    """
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    cols = ["StudentID", "Week", "StudyHours", "Attendance", "AssignmentsCompleted", "Prediction", "Timestamp"]
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
    else:
        df = pd.DataFrame(columns=cols)

    week = int(week)

    # Remove existing record for StudentID+Week (update behavior)
    df = df[~((df["StudentID"] == student_id) & (df["Week"] == week))]

    new_row = {
        "StudentID": student_id,
        "Week": week,
        "StudyHours": int(study_hours),
        "Attendance": int(attendance),
        "AssignmentsCompleted": int(assignments),
        "Prediction": prediction,
        "Timestamp": datetime.now().isoformat(timespec="seconds")
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)
    return True

# -------------------------
# Read logs helper
# -------------------------
def read_logs():
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        # ensure types
        df["Week"] = pd.to_numeric(df["Week"], errors="coerce").fillna(0).astype(int)
        return df
    return pd.DataFrame(columns=["StudentID", "Week", "StudyHours", "Attendance", "AssignmentsCompleted", "Prediction", "Timestamp"])

# -------------------------
# Report generation (PDF + chart)
# -------------------------
def generate_report(student_id):
    """
    Generates a PDF report for student_id containing a table of logs and a small trend chart.
    Returns path to PDF file.
    """
    df = read_logs()
    student_df = df[df["StudentID"] == student_id].sort_values("Week")

    if student_df.empty:
        raise ValueError("No logs found for this student.")

    # Map predictions to numeric scores for a simple trend (High=3, Medium=2, Low=1)
    mapping = {"Low": 1, "Medium": 2, "High": 3}
    student_df["PredScore"] = student_df["Prediction"].map(mapping).fillna(0).astype(int)

    # Create a small matplotlib chart (week vs PredScore)
    plt.figure(figsize=(6, 3))
    plt.plot(student_df["Week"], student_df["PredScore"], marker="o")
    plt.xticks(student_df["Week"].tolist())
    plt.yticks([1,2,3], ["Low","Medium","High"])
    plt.title(f"Performance Trend - {student_id}")
    plt.xlabel("Week")
    plt.ylabel("Prediction")
    plt.grid(alpha=0.3)
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    # Create PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, f"Performance Report - {student_id}", ln=True, align="C")
    pdf.ln(6)

    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 6, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.ln(4)

    # Insert chart image
    img_path = f"temp_chart_{student_id}.png"
    with open(img_path, "wb") as f:
        f.write(buf.getbuffer())

    pdf.image(img_path, x=15, w=180)
    os.remove(img_path)
    pdf.ln(6)

    # Add a small table of logs
    pdf.set_font("Arial", "B", 11)
    pdf.cell(30, 6, "Week", 1)
    pdf.cell(30, 6, "StudyHrs", 1)
    pdf.cell(30, 6, "Attend", 1)
    pdf.cell(40, 6, "Assignments", 1)
    pdf.cell(40, 6, "Prediction", 1)
    pdf.ln()
    pdf.set_font("Arial", size=10)
    for _, row in student_df.iterrows():
        pdf.cell(30, 6, str(row["Week"]), 1)
        pdf.cell(30, 6, str(row["StudyHours"]), 1)
        pdf.cell(30, 6, str(row["Attendance"]), 1)
        pdf.cell(40, 6, str(row["AssignmentsCompleted"]), 1)
        pdf.cell(40, 6, str(row["Prediction"]), 1)
        pdf.ln()

    out_path = f"report_{student_id}.pdf"
    pdf.output(out_path)
    return out_path

# -------------------------
# Chatbot wrapper (OpenAI fallback -> simple heuristic)
# -------------------------
def chatbot_answer(query):
    """
    If OPENAI_API_KEY is available, call OpenAI ChatCompletion (gpt-3.5-turbo).
    Otherwise return a simple rule-based friendly answer.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Simple helpful fallback
        q = query.lower()
        if "study plan" in q or "plan" in q:
            return ("Try this small plan: 1) Study 1 hour focused daily (Pomodoro 25/5). "
                    "2) Weekly revision on Sundays. 3) Do 5 practice problems daily.")
        if "motivate" in q or "motivation" in q:
            return "Small consistent steps matter. Celebrate small wins and keep a weekly goal."
        return ("Chatbot offline (no API key). Quick tip: track study hours, attend classes, "
                "complete assignments. For a detailed reply, set OPENAI_API_KEY in your environment.")
    try:
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful, encouraging academic mentor."},
                {"role": "user", "content": query}
            ],
            max_tokens=300,
            temperature=0.6
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error contacting chat API: {e}"
