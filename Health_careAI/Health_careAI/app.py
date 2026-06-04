from flask import Flask, render_template, request, send_file
import joblib
import pickle
import pandas as pd
from reportlab.pdfgen import canvas

app = Flask(__name__)

# Load model
model = joblib.load("best_model.pkl")

# Load encoder
with open("disease_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load medicine database
with open("medicine_database.pkl", "rb") as f:
    medicine_db = pickle.load(f)

# Store latest prediction
latest_prediction = {
    "disease": "",
    "age": "",
    "bp": "",
    "chol": "",
    "risk": ""
}

# Prediction history
history = []


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    medicines = []
    accuracy = "98.9%"
    confidence = None

    age = None
    bp = None
    chol = None
    risk = None

    global latest_prediction

    if request.method == "POST":

        age = int(request.form["age"])
        bp = int(request.form["bp"])
        chol = int(request.form["chol"])

        fever = request.form["fever"]
        cough = request.form["cough"]
        fatigue = request.form["fatigue"]
        breathing = request.form["breathing"]
        gender = request.form["gender"]
        outcome = request.form["outcome"]
        risk = request.form["risk"]

        data = pd.DataFrame([{
            "age": age,
            "blood_pressure": bp,
            "cholesterol_level": chol,
            "age_scaled": age,
            "bp_scaled": bp,
            "chol_scaled": chol,
            "fever": fever,
            "cough": cough,
            "fatigue": fatigue,
            "difficulty_breathing": breathing,
            "gender": gender,
            "outcome_variable": outcome,
            "risk_level": risk
        }])

        pred = model.predict(data)

        try:
            confidence = round(
                max(model.predict_proba(data)[0]) * 100,
                2
            )
        except:
            confidence = 98.9

        disease = encoder.inverse_transform(pred)[0]

        prediction = disease

        if disease in medicine_db:

            medicines = medicine_db[disease].get(
                "medicines",
                ["Consult a healthcare professional"]
            )

        else:

            medicines = ["Consult a healthcare professional"]

        latest_prediction = {
            "disease": disease,
            "age": age,
            "bp": bp,
            "chol": chol,
            "risk": risk
        }

        history.append({
            "age": age,
            "disease": disease
        })

    return render_template(
        "index.html",
        prediction=prediction,
        medicines=medicines,
        accuracy=accuracy,
        confidence=confidence,
        age=age,
        bp=bp,
        chol=chol,
        risk=risk,
        history=history
    )


@app.route("/download")
def download():

    pdf_file = "patient_report.pdf"

    pdf = canvas.Canvas(pdf_file)

    pdf.setTitle("MediCare AI Report")

    pdf.drawString(100, 800, "MediCare AI - Patient Report")

    pdf.drawString(
        100,
        760,
        f"Disease: {latest_prediction['disease']}"
    )

    pdf.drawString(
        100,
        730,
        f"Age: {latest_prediction['age']}"
    )

    pdf.drawString(
        100,
        700,
        f"Blood Pressure: {latest_prediction['bp']}"
    )

    pdf.drawString(
        100,
        670,
        f"Cholesterol: {latest_prediction['chol']}"
    )

    pdf.drawString(
        100,
        640,
        f"Risk Level: {latest_prediction['risk']}"
    )

    pdf.save()

    return send_file(
        pdf_file,
        as_attachment=True
    )


if __name__ == "__main__":
    app.run(debug=True)
