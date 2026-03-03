
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint
import joblib
import numpy as np
import logging
from scipy.stats import ks_2samp
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------
logging.basicConfig(
    filename="drift_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------------
# Drift Monitoring Module
# ------------------------------------------------------------------
class DriftMonitor:
    def __init__(self, reference_path: str, psi_threshold: float = 0.25):
        try:
            self.reference = joblib.load(reference_path)
        except Exception:
            self.reference = None
        self.psi_threshold = psi_threshold

    def calculate_psi(self, expected, actual, bins=10):
        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)

        expected_perc = expected_counts / (len(expected) + 1e-6)
        actual_perc = actual_counts / (len(actual) + 1e-6)

        psi = np.sum(
            (actual_perc - expected_perc)
            * np.log((actual_perc + 1e-6) / (expected_perc + 1e-6))
        )
        return psi

    def check_drift(self, train_feature, new_feature):
        psi_value = self.calculate_psi(train_feature, new_feature)
        _, p_value = ks_2samp(train_feature, new_feature)

        drift_detected = psi_value > self.psi_threshold or p_value < 0.05

        return {
            "psi": float(psi_value),
            "p_value": float(p_value),
            "drift_detected": drift_detected
        }


# ------------------------------------------------------------------
# FastAPI App Initialization
# ------------------------------------------------------------------
app = FastAPI(
    title="DASS-42 Mental Health API with Drift Monitoring",
    version="2.0"
)

MODEL_PATH = "final_dass_system.pkl"
REFERENCE_PATH = "reference_stats.pkl"

try:
    bundle = joblib.load(MODEL_PATH)
except FileNotFoundError:
    bundle = None

drift_monitor = DriftMonitor(REFERENCE_PATH)


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------
class DASSAssessment(BaseModel):
    answers: List[conint(ge=0, le=3)] = Field(..., min_length=42, max_length=42)
    education: int
    urban: int
    gender: int
    engnat: int
    age: int
    screensize: float
    religion: int
    orientation: int
    race: int
    married: int
    familysize: int


class DASSResponse(BaseModel):
    depression_level: int
    anxiety_level: int
    stress_level: int
    drift_detected: bool


SEVERITY_MAPPING = {
    0: "Normal",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Extremely Severe"
}

DEP_IDX = [2, 4, 9, 12, 15, 16, 20, 23, 25, 30, 33, 36, 37, 41]
ANX_IDX = [1, 3, 6, 8, 14, 18, 19, 22, 24, 27, 29, 35, 39, 40]
STR_IDX = [0, 5, 7, 10, 11, 13, 17, 21, 26, 28, 31, 32, 34, 38]


# ------------------------------------------------------------------
# Prediction Endpoint with Drift Check
# ------------------------------------------------------------------
@app.post("/predict", response_model=DASSResponse)
def predict(payload: DASSAssessment):

    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        dv = [
            payload.education, payload.urban, payload.gender, payload.engnat,
            payload.age, payload.screensize, payload.religion, payload.orientation,
            payload.race, payload.married, payload.familysize
        ]

        dep_answers = [payload.answers[i] for i in DEP_IDX]
        anx_answers = [payload.answers[i] for i in ANX_IDX]
        str_answers = [payload.answers[i] for i in STR_IDX]

        dep_feat = np.array([dep_answers + dv])
        anx_feat = np.array([anx_answers + dv])
        str_feat = np.array([str_answers + dv])

        dep_scaled = bundle["dep_scaler"].transform(dep_feat)
        anx_scaled = bundle["anx_scaler"].transform(anx_feat)
        str_scaled = bundle["str_scaler"].transform(str_feat)

        dep_level = int(bundle["depression_model"].predict(dep_scaled)[0])
        anx_level = int(bundle["anxiety_model"].predict(anx_scaled)[0])
        str_level = int(bundle["stress_model"].predict(str_scaled)[0])

        # ---------------- Drift Monitoring ----------------
        drift_flag = False
        if drift_monitor.reference:
            ref_mean = drift_monitor.reference.get("mean", None)
            if ref_mean is not None:
                drift_result = drift_monitor.check_drift(
                    ref_mean, dep_feat.flatten()
                )
                drift_flag = drift_result["drift_detected"]

                if drift_flag:
                    logging.warning(f"Drift detected: {drift_result}")
                else:
                    logging.info(f"No drift: {drift_result}")

        return DASSResponse(
            depression_level=dep_level,
            anxiety_level=anx_level,
            stress_level=str_level,
            drift_detected=drift_flag
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def health():
    return {"status": "Running with Drift Monitoring"}
