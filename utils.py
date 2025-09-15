
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from fpdf import FPDF
from io import BytesIO
from transformers import pipeline

# Paths
MODEL_PATH = "saved_model.pkl"
LOG_PATH = "performance_log.csv"

# Hugging Face pipeline for chatbot and intervention plan
hf_chatbot = pipeline("text2text-generation", model="google/flan-t5-base")

def ask_hf_system(user_prompt: str, system_prompt: str = "You are a helpful educational assistant.") -> str:
    """
    Generates concise, actionable responses using Hugging Face.
    """
    try:
        prompt = f"{system_prompt}\nUser: {user_prompt}\nAssistant:"
        response = hf_chatbot(prompt, max_length=300, do_sample=True)
        return response[0]['generated_text'].strip()
    except Exception as e:
        return f"⚠️ Hugging Face error: {e}"

# -------------------------
# Logging
# -------------------------
def log_prediction(student_id: str, week: int, row_dict: dict, predicted_g3: float):
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    cols = ["StudentID", "Week"] + list(row_dict.keys()) + ["Predicted_G3", "Timestamp"]

    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
    else:
        df = pd.DataFrame(columns=cols)

    week = int(week)
    df = df[~((df["StudentID"] == student_id) & (df["Week"] == week))]

    new_row = {"StudentID": student_id, "Week": week}
    new_row.update(row_dict)
    new_row["Predicted_G3"] = float(predicted_g3)
    new_row["Timestamp"] = datetime.now().isoformat(timespec="seconds")

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)
    return True

def read_logs():
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame()

# -------------------------
# PDF Report
# -------------------------
def generate_pdf_report(student_id: str):
    df = read_logs()
    student_df = df[df["StudentID"] == student_id].sort_values("Week")
    if student_df.empty:
        raise ValueError("No logs for this student.")

    # Chart
    plt.figure(figsize=(6, 3))
    plt.plot(student_df["Week"], student_df["Predicted_G3"], marker="o", linewidth=2)
    plt.xlabel("Week")
    plt.ylabel("Predicted G3")
    plt.title(f"Predicted G3 Trend - {student_id}")
    plt.grid(alpha=0.3)
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, f"Performance Report - {student_id}", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 6, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.ln(6)

    img_temp = f"temp_{student_id}.png"
    with open(img_temp, "wb") as f:
        f.write(buf.getbuffer())
    pdf.image(img_temp, x=15, w=180)
    os.remove(img_temp)
    pdf.ln(6)

    pdf.set_font("Arial", "B", 10)
    headers = ["Week", "Predicted_G3"]
    for h in headers:
        pdf.cell(40, 7, h, 1)
    pdf.ln()
    pdf.set_font("Arial", size=10)
    for _, r in student_df.iterrows():
        pdf.cell(40, 7, str(int(r["Week"])), 1)
        pdf.cell(40, 7, f"{float(r['Predicted_G3']):.2f}", 1)
        pdf.ln()

    out_path = f"report_{student_id}.pdf"
    pdf.output(out_path)
    return out_path

# -------------------------
# Intervention plan
# -------------------------
def generate_intervention_plan(student_profile: dict, predicted_g3: float) -> str:
    profile_text = "\n".join([f"{k}: {v}" for k, v in student_profile.items()])
    prompt = (
        f"You are an empathetic academic mentor.\n"
        f"Student profile:\n{profile_text}\n"
        f"Predicted final grade (G3): {predicted_g3:.2f}/20\n\n"
        "Provide a concise, actionable 2-week study plan with daily goals, weekly targets, "
        "specific steps, and motivational tips. Make it practical and easy to follow."
    )
    return ask_hf_system(prompt)


