from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint
import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore")

app = FastAPI(
    title="DASS-42 Mental Health API",
    description="A concise and robust API for predicting Depression, Anxiety, and Stress levels using a Pickled ML model.",
    version="1.0"
)


MODEL_PATH = "final_dass_system.pkl"

print("\n🔄 Loading DASS Mental Health Models...")
try:
    bundle = joblib.load(MODEL_PATH)
    print("✅ All 3 models loaded: Depression | Anxiety | Stress")
except FileNotFoundError:
    print(f"❌ '{MODEL_PATH}' not found. Server may crash at prediction.")
    bundle = None

class DASSAssessmentBase(BaseModel):
    # Questions (42 answers in 0-3 scale, mapped exactly to chronological DASS42)
    answers: List[conint(ge=0, le=3)] = Field(
        ...,
        description="List of 42 answers in the order of DASS-42 (Index 0 = Q1, Index 41 = Q42)",
        min_length=42,
        max_length=42,
    )

    # Demographic data (11 features needed for normalization as found in the baseline modeling)
    education: int = Field(..., description="1=Less than high school, 2=High school, 3=University, 4=Graduate degree")
    urban: int = Field(..., description="1=Rural, 2=Suburban, 3=Urban")
    gender: int = Field(..., description="1=Male, 2=Female, 3=Other")
    engnat: int = Field(..., description="English native? 1=Yes, 2=No")
    age: int = Field(..., description="Age in years")
    screensize: float = Field(..., description="Device screen size in inches")
    religion: int = Field(..., description="Religion classification (1-10)")
    orientation: int = Field(..., description="Sexual orientation (1-5)")
    race: int = Field(..., description="Race category (10=Asian, ..., 60=White, etc.)")
    married: int = Field(..., description="Marital status (1=Never, 2=Married, 3=Prev. married)")
    familysize: int = Field(..., description="Number of family members")

class DASSResponse(BaseModel):
    depression_level: int
    depression_severity: str
    anxiety_level: int
    anxiety_severity: str
    stress_level: int
    stress_severity: str

SEVERITY_MAPPING = {
    0: "Normal",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Extremely Severe"
}

# The expected index mappings for each subscale based on chronological DASS42 ordering
DEP_IDX = [2, 4, 9, 12, 15, 16, 20, 23, 25, 30, 33, 36, 37, 41]  # 0-indexed values
ANX_IDX = [1, 3, 6, 8, 14, 18, 19, 22, 24, 27, 29, 35, 39, 40]
STR_IDX = [0, 5, 7, 10, 11, 13, 17, 21, 26, 28, 31, 32, 34, 38]


@app.post("/predict", response_model=DASSResponse, summary="Perform DASS-42 Prediction")
def predict_dass(payload: DASSAssessmentBase):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    if payload is None:
        raise HTTPException(status_code=400, detail="Payload is required.")

    try:
        # Common demographic values string
        dv = [
            payload.education, payload.urban, payload.gender, payload.engnat,
            payload.age, payload.screensize, payload.religion, payload.orientation,
            payload.race, payload.married, payload.familysize
        ]

        # Extract subscale vectors
        dep_answers = [payload.answers[i] for i in DEP_IDX]
        anx_answers = [payload.answers[i] for i in ANX_IDX]
        str_answers = [payload.answers[i] for i in STR_IDX]

        dep_feat = np.array([dep_answers + dv])
        anx_feat = np.array([anx_answers + dv])
        str_feat = np.array([str_answers + dv])

        # Feature scaling (Inherently computes Z-scores via StandardScaler)
        dep_scaled = bundle["dep_scaler"].transform(dep_feat)
        anx_scaled = bundle["anx_scaler"].transform(anx_feat)
        str_scaled = bundle["str_scaler"].transform(str_feat)

        # Predicting
        dep_level = int(bundle["depression_model"].predict(dep_scaled)[0])
        anx_level = int(bundle["anxiety_model"].predict(anx_scaled)[0])
        str_level = int(bundle["stress_model"].predict(str_scaled)[0])

        return DASSResponse(
            depression_level=dep_level,
            depression_severity=SEVERITY_MAPPING.get(dep_level, "Unknown"),
            anxiety_level=anx_level,
            anxiety_severity=SEVERITY_MAPPING.get(anx_level, "Unknown"),
            stress_level=str_level,
            stress_severity=SEVERITY_MAPPING.get(str_level, "Unknown")
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/", summary="Health Check")
def root():
    return {
        "service": "DASS-42 Mental Health API",
        "status": "Online" if bundle else "Degraded - Model Not Found"
    }
