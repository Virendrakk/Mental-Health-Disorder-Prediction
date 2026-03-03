from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import joblib
from datetime import datetime

# ---------------------------------------------------
# Initialize App
# ---------------------------------------------------
app = FastAPI(
    title="DASS Mental Health Assessment API",
    description="Telehealth API for Depression, Anxiety & Stress Prediction",
    version="1.0"
)

# ---------------------------------------------------
# Load Saved Artifacts
# ---------------------------------------------------
dass_system = joblib.load("final_dass_system.pkl")
training_stats = joblib.load("training_stats.pkl")
dep_model = dass_system["depression_model"]
anx_model = dass_system["anxiety_model"]
str_model = dass_system["stress_model"]

dep_scaler = dass_system["dep_scaler"]
anx_scaler = dass_system["anx_scaler"]
str_scaler = dass_system["str_scaler"]


# ---------------------------------------------------
# Question Schemas
# ---------------------------------------------------
class DepressionQuestions(BaseModel):
    Q3A: int = Field(description="I couldn't seem to experience any positive feeling at all", ge=0, le=3)
    Q5A: int = Field(description="I felt down-hearted and blue", ge=0, le=3)
    Q10A: int = Field(description="I couldn't seem to get any enjoyment from my daily activities", ge=0, le=3)
    Q13A: int = Field(description="I felt that I was not worth much as a person", ge=0, le=3)
    Q16A: int = Field(description="I felt that life was not worth living", ge=0, le=3)
    Q17A: int = Field(description="I felt I was a failure as a person", ge=0, le=3)
    Q21A: int = Field(description="I couldn't seem to feel anything", ge=0, le=3)
    Q24A: int = Field(description="I felt hopeless", ge=0, le=3)
    Q26A: int = Field(description="I felt I was trapped", ge=0, le=3)
    Q31A: int = Field(description="I felt that I was a burden to others", ge=0, le=3)
    Q34A: int = Field(description="I felt I was not doing well in life", ge=0, le=3)
    Q37A: int = Field(description="I felt I was not achieving anything", ge=0, le=3)
    Q38A: int = Field(description="I felt I was not progressing in life", ge=0, le=3)
    Q42A: int = Field(description="I felt I was not making any progress in life", ge=0, le=3)


class AnxietyQuestions(BaseModel):
    Q2A: int = Field(description="I felt nervous and restless", ge=0, le=3)
    Q4A: int = Field(description="I felt I was on edge", ge=0, le=3)
    Q7A: int = Field(description="I couldn't seem to stop worrying about anything", ge=0, le=3)
    Q9A: int = Field(description="I worried too much about things I should not worry about", ge=0, le=3)
    Q15A: int = Field(description="I felt afraid without any good reason", ge=0, le=3)
    Q19A: int = Field(description="I felt that I was not safe", ge=0, le=3)
    Q20A: int = Field(description="I felt that I was not in control of my life", ge=0, le=3)
    Q23A: int = Field(description="I felt that I was not able to cope with things", ge=0, le=3)
    Q25A: int = Field(description="I felt overwhelmed by things I had to do", ge=0, le=3)
    Q28A: int = Field(description="I felt that I was not doing well in life", ge=0, le=3)
    Q30A: int = Field(description="I felt that I was not achieving anything", ge=0, le=3)
    Q36A: int = Field(description="I felt that I was not progressing in life", ge=0, le=3)
    Q40A: int = Field(description="I felt I was not making any progress in life", ge=0, le=3)
    Q41A: int = Field(description="I felt that my life was not worth living", ge=0, le=3)


class StressQuestions(BaseModel):
    Q1A: int = Field(description="I felt that I was under a lot of pressure", ge=0, le=3)
    Q6A: int = Field(description="I felt that I was not able to cope with things", ge=0, le=3)
    Q8A: int = Field(description="I felt that I was not in control of my life", ge=0, le=3)
    Q11A: int = Field(description="I felt overwhelmed by things I had to do", ge=0, le=3)
    Q12A: int = Field(description="I felt that I was not doing well in life", ge=0, le=3)
    Q14A: int = Field(description="I felt that I was not achieving anything", ge=0, le=3)
    Q18A: int = Field(description="I felt that my life was not worth living", ge=0, le=3)
    Q22A: int = Field(description="I felt that I was not making any progress in life", ge=0, le=3)
    Q27A: int = Field(description="I felt that I was not able to do anything right", ge=0, le=3)
    Q29A: int = Field(description="I felt that I was not getting anywhere in life", ge=0, le=3)
    Q32A: int = Field(description="I felt that I was not achieving my goals", ge=0, le=3)
    Q33A: int = Field(description="I felt that I was not living up to my potential", ge=0, le=3)
    Q35A: int = Field(description="I felt that I was not making any sense of my life", ge=0, le=3)
    Q39A: int = Field(description="I felt that my life had no meaning", ge=0, le=3)


class DASSInput(BaseModel):
    depression: DepressionQuestions
    anxiety: AnxietyQuestions
    stress: StressQuestions


# ---------------------------------------------------
# Drift Check
# ---------------------------------------------------
def check_drift(model_key, new_data):

    train_mean = training_stats[model_key]["mean"]
    train_std = training_stats[model_key]["std"]

    z_score = np.abs((new_data - train_mean) / (train_std + 1e-8))
    drift_flag = (z_score > 3).any()

    return bool(drift_flag)


# ---------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------
@app.post("/predict")
def predict_dass(data: DASSInput):

    # Combine all 42 questions in SAME order used during training
    input_list = list(data.depression.dict().values()) + \
                 list(data.anxiety.dict().values()) + \
                 list(data.stress.dict().values())

    input_array = np.array(input_list).reshape(1, -1)

    # Scale
    input_scaled = dep_scaler.transform(input_array)

    # Predictions
    dep_pred = dep_model.predict(input_scaled)[0]
    anx_pred = anx_model.predict(input_scaled)[0]
    str_pred = str_model.predict(input_scaled)[0]

    # Drift Check
    dep_drift = check_drift("depression", input_array)
    anx_drift = check_drift("anxiety", input_array)
    str_drift = check_drift("stress", input_array)

    overall_drift = dep_drift or anx_drift or str_drift

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": "DASS_v1.0",
        "Depression_Level": int(dep_pred),
        "Anxiety_Level": int(anx_pred),
        "Stress_Level": int(str_pred),
        "Drift_Detected": overall_drift
    }


# ---------------------------------------------------
# Health Check
# ---------------------------------------------------
@app.get("/")
def health():
    return {"status": "API Running Successfully"}