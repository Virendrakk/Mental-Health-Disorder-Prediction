"""
╔══════════════════════════════════════════════════════════════════╗
║         DASS-42 Mental Health Prediction API                     ║
║    Depression · Anxiety · Stress Assessment System              ║
║    Built for: Telehealth Apps & Wellness Platforms              ║
╚══════════════════════════════════════════════════════════════════╝

HOW TO START:
  pip install fastapi uvicorn joblib scikit-learn pydantic numpy
  uvicorn main:app --reload --host 0.0.0.0 --port 8000

  Then open: http://localhost:8000/docs  ← Interactive test page!

ANSWER SCALE (all 42 questions):
  0 = Did not apply to me at all
  1 = Applied to me to some degree / some of the time
  2 = Applied to me to a considerable degree / good part of time
  3 = Applied to me very much / most of the time
"""

import time
import warnings
import numpy as np
from datetime import datetime
from typing import Optional

warnings.filterwarnings("ignore")
import joblib

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════
#  SECTION 1 — LOAD THE DASS MODEL BUNDLE
#  Done ONCE at server startup — not on every request
# ══════════════════════════════════════════════════════════════════

MODEL_PATH = "final_dass_system.pkl"

print("\n🔄 Loading DASS Mental Health Models...")
try:
    bundle           = joblib.load(MODEL_PATH)
    depression_model = bundle["depression_model"]
    anxiety_model    = bundle["anxiety_model"]
    stress_model     = bundle["stress_model"]
    dep_scaler       = bundle["dep_scaler"]
    anx_scaler       = bundle["anx_scaler"]
    str_scaler       = bundle["str_scaler"]
    print("✅ All 3 models loaded: Depression | Anxiety | Stress")
except FileNotFoundError:
    print(f"❌ '{MODEL_PATH}' not found. Place it in the same folder and restart.")
    bundle = None


# ══════════════════════════════════════════════════════════════════
#  SECTION 2 — SEVERITY LABELS, COLORS & WELLNESS TIPS
# ══════════════════════════════════════════════════════════════════

SEVERITY_LABELS = {
    0: "Normal",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Extremely Severe"
}

SEVERITY_COLORS = {
    0: "green",
    1: "yellow",
    2: "orange",
    3: "red",
    4: "dark_red"
}

WELLNESS_TIPS = {
    "depression": {
        0: "You appear to be coping well emotionally. Keep up healthy routines like regular sleep, exercise, and social connection.",
        1: "Mild low mood detected. Try light exercise, journaling, and talking to a trusted friend. Monitor over the next 2 weeks.",
        2: "Moderate depression symptoms present. We recommend speaking with a counsellor or therapist. Self-care alone may not be sufficient.",
        3: "Severe depression indicators. Please consult a mental health professional promptly. Do not manage this alone.",
        4: "Extremely severe depression. Immediate professional support is strongly advised. If you are having thoughts of self-harm, please contact a crisis helpline now."
    },
    "anxiety": {
        0: "Anxiety levels appear normal. Maintain stress management habits like breathing exercises and mindfulness.",
        1: "Mild anxiety noted. Try diaphragmatic breathing, reducing caffeine, and short mindfulness sessions daily.",
        2: "Moderate anxiety present. Consider guided therapy (CBT works well for anxiety). Limit news and social media intake.",
        3: "Severe anxiety detected. Professional evaluation is recommended. Medication assessment may be appropriate.",
        4: "Extremely severe anxiety. Please seek urgent mental health support. Avoid self-medicating."
    },
    "stress": {
        0: "Stress levels are within healthy range. Continue current coping strategies.",
        1: "Mild stress noted. Review your workload and priorities. Build in regular breaks and physical activity.",
        2: "Moderate stress present. Consider a structured stress management programme. Talk to someone you trust.",
        3: "Severe stress detected. This level is unsustainable long-term. Consult a health professional and reduce major stressors where possible.",
        4: "Extremely severe stress. Immediate lifestyle review and professional support are strongly recommended."
    }
}


def get_urgency_flag(dep_cls: int, anx_cls: int, str_cls: int) -> dict:
    """
    Tells the telehealth provider how urgently to follow up.
    Based on the highest severity score across all 3 conditions.
    """
    max_sev = max(dep_cls, anx_cls, str_cls)
    if max_sev == 4:
        return {
            "level"  : "URGENT",
            "color"  : "red",
            "message": "Extremely severe score detected. Provider follow-up recommended within 24 hours."
        }
    elif max_sev == 3:
        return {
            "level"  : "HIGH",
            "color"  : "orange",
            "message": "Severe score in at least one domain. Recommend scheduling within 48–72 hours."
        }
    elif max_sev == 2:
        return {
            "level"  : "MODERATE",
            "color"  : "yellow",
            "message": "Moderate symptoms detected. Recommend check-in within 1–2 weeks."
        }
    else:
        return {
            "level"  : "ROUTINE",
            "color"  : "green",
            "message": "Scores within normal to mild range. Standard monitoring applies."
        }


# ══════════════════════════════════════════════════════════════════
#  SECTION 3 — CREATE THE FASTAPI APP
# ══════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "DASS-42 Mental Health Assessment API",
    description = """
## 🧠 DASS-42 Mental Health Prediction API
### Built for Telehealth Apps & Wellness Platforms

Processes DASS-42 questionnaire responses and returns clinical severity
classifications for **Depression**, **Anxiety**, and **Stress**.

---

### How it works:
1. User completes the DASS-42 questionnaire in your app
2. Your app calls **POST /assess** with all 42 answers + demographics
3. API returns severity levels, wellness tips, and urgency flag instantly

### Answer scale (0–3 for all questions):
| Score | Meaning |
|-------|---------|
| **0** | Did not apply at all |
| **1** | Applied sometimes |
| **2** | Applied often |
| **3** | Applied most of the time |

### Severity levels returned per condition:
`Normal` → `Mild` → `Moderate` → `Severe` → `Extremely Severe`
    """,
    version="1.0.0"
)

# Allow your frontend/mobile app to call this API from any domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production: replace * with your app's domain
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════
#  SECTION 4 — INPUT DATA MODELS (The "forms" sent to the API)
# ══════════════════════════════════════════════════════════════════

def q(desc: str, ex: int = 1):
    """Shortcut for creating a DASS question field (must be 0, 1, 2, or 3)."""
    return Field(..., ge=0, le=3, description=desc, example=ex)


class DemographicInfo(BaseModel):
    """
    Basic user demographics — used alongside question answers by the model.
    These 11 fields are required by all 3 sub-models.
    """
    age        : int   = Field(..., ge=10, le=100, description="Age in years", example=28)
    gender     : int   = Field(..., ge=1, le=3,   description="1=Male  2=Female  3=Other", example=2)
    education  : int   = Field(..., ge=1, le=4,   description="1=Less than high school  2=High school  3=University  4=Graduate degree", example=3)
    urban      : int   = Field(..., ge=1, le=3,   description="1=Rural  2=Suburban  3=Urban", example=2)
    engnat     : int   = Field(..., ge=1, le=2,   description="Is English your native language? 1=Yes  2=No", example=1)
    screensize : float = Field(...,               description="Device screen size in inches (e.g. 6.1=phone, 15=laptop)", example=6.1)
    religion   : int   = Field(..., ge=1, le=10,  description="1=Agnostic 2=Atheist 3=Buddhist 4=Christian 5=Hindu 6=Jewish 7=Muslim 8=Sikh 9=Other 10=Unknown", example=4)
    orientation: int   = Field(..., ge=1, le=5,   description="1=Heterosexual 2=Bisexual 3=Homosexual 4=Asexual 5=Other", example=1)
    race       : int   = Field(...,               description="10=Asian 20=Arab 30=Black 40=Indigenous 50=Hispanic 60=White 70=Other", example=60)
    married    : int   = Field(..., ge=1, le=3,   description="1=Never married  2=Currently married  3=Previously married", example=1)
    familysize : int   = Field(..., ge=1, le=20,  description="Number of people in household", example=4)


class DepressionQuestions(BaseModel):
    """14 Depression subscale questions from the DASS-42 (Q3A to Q42A)."""
    Q3A : int = q("I couldn't seem to experience any positive feeling at all")
    Q5A : int = q("I just couldn't seem to get going")
    Q10A: int = q("I felt that I had nothing to look forward to")
    Q13A: int = q("I felt sad and depressed")
    Q16A: int = q("I felt that I had lost interest in just about everything")
    Q17A: int = q("I felt I wasn't worth much as a person")
    Q21A: int = q("I felt that life was meaningless")
    Q24A: int = q("I couldn't seem to get any enjoyment out of the things I did")
    Q26A: int = q("I felt down-hearted and blue")
    Q31A: int = q("I was unable to become enthusiastic about anything")
    Q34A: int = q("I felt I was pretty worthless")
    Q37A: int = q("I felt that life wasn't worthwhile")
    Q38A: int = q("I could see nothing in the future to be hopeful about")
    Q42A: int = q("I found it difficult to work up the initiative to do things")


class AnxietyQuestions(BaseModel):
    """14 Anxiety subscale questions from the DASS-42 (Q2A to Q41A)."""
    Q2A : int = q("I was aware of dryness of my mouth", 0)
    Q4A : int = q("I experienced breathing difficulty", 0)
    Q7A : int = q("I experienced trembling (e.g., in the hands)", 0)
    Q9A : int = q("I was worried about situations in which I might panic", 1)
    Q15A: int = q("I felt I was close to panic", 0)
    Q19A: int = q("I was aware of the action of my heart in the absence of physical exertion", 0)
    Q20A: int = q("I felt scared without any good reason", 1)
    Q23A: int = q("I had difficulty swallowing", 0)
    Q25A: int = q("I was aware of changes in my heart rate", 0)
    Q28A: int = q("I felt I was about to have a panic attack", 0)
    Q30A: int = q("I felt terror", 0)
    Q36A: int = q("I was scared without any good reason", 0)
    Q40A: int = q("I felt my heart beating fast", 1)
    Q41A: int = q("I was afraid that I would be thrown by some trivial but unfamiliar task", 0)


class StressQuestions(BaseModel):
    """14 Stress subscale questions from the DASS-42 (Q1A to Q39A)."""
    Q1A : int = q("I found it hard to wind down", 2)
    Q6A : int = q("I tended to over-react to situations", 1)
    Q8A : int = q("I felt that I was using a lot of nervous energy", 2)
    Q11A: int = q("I found myself getting agitated", 2)
    Q12A: int = q("I found it difficult to relax", 2)
    Q14A: int = q("I was intolerant of anything that kept me from getting on with what I was doing", 1)
    Q18A: int = q("I felt that I was rather touchy", 1)
    Q22A: int = q("I found it hard to calm down after something upset me", 1)
    Q27A: int = q("I was irritable", 2)
    Q29A: int = q("I found it hard to calm down", 1)
    Q32A: int = q("I found it difficult to tolerate interruptions to what I was doing", 1)
    Q33A: int = q("I was in a state of nervous tension", 2)
    Q35A: int = q("I felt that demands were building so high that I could not overcome them", 1)
    Q39A: int = q("I became irritated when things didn't go my way", 1)


class DASSRequest(BaseModel):
    """
    Full DASS-42 Assessment Request.
    Groups all 42 questions into 3 sections + demographics.
    Optionally include user_id and session_id for your platform's records.
    """
    depression  : DepressionQuestions
    anxiety     : AnxietyQuestions
    stress      : StressQuestions
    demographics: DemographicInfo
    user_id     : Optional[str] = Field(None, description="Your platform's user ID (optional)", example="user_abc123")
    session_id  : Optional[str] = Field(None, description="Session ID for audit trail (optional)", example="session_xyz789")


# ══════════════════════════════════════════════════════════════════
#  SECTION 5 — CORE PREDICTION LOGIC
# ══════════════════════════════════════════════════════════════════

def run_prediction(request: DASSRequest) -> dict:
    """
    Runs all 3 DASS predictions (Depression, Anxiety, Stress).

    Plain English steps:
      1. Pull answers in the EXACT feature order each model was trained on
      2. Add demographic fields in the correct order
      3. Scale (normalise) the values using each model's own scaler
      4. Each model predicts a severity class (0–4) + probabilities
      5. Map numbers → human labels, add tips and urgency flag
    """
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Place final_dass_system.pkl next to main.py and restart."
        )

    d   = request.depression
    a   = request.anxiety
    s   = request.stress
    dem = request.demographics

    # Demographic values — shared by all 3 models (must be in this exact order)
    dv = [
        dem.education, dem.urban, dem.gender, dem.engnat,
        dem.age, dem.screensize, dem.religion, dem.orientation,
        dem.race, dem.married, dem.familysize
    ]

    # ── Depression Model ────────────────────────────────────────────
    # Feature order: Q3A, Q5A, Q10A, Q13A, Q16A, Q17A, Q21A, Q24A,
    #                Q26A, Q31A, Q34A, Q37A, Q38A, Q42A, + 11 demographics
    dep_feat  = np.array([[
        d.Q3A, d.Q5A, d.Q10A, d.Q13A, d.Q16A, d.Q17A, d.Q21A,
        d.Q24A, d.Q26A, d.Q31A, d.Q34A, d.Q37A, d.Q38A, d.Q42A,
        *dv
    ]])
    dep_scaled = dep_scaler.transform(dep_feat)
    dep_cls    = int(depression_model.predict(dep_scaled)[0])
    dep_proba  = depression_model.predict_proba(dep_scaled)[0]

    # ── Anxiety Model ───────────────────────────────────────────────
    # Feature order: Q2A, Q4A, Q7A, Q9A, Q15A, Q19A, Q20A, Q23A,
    #                Q25A, Q28A, Q30A, Q36A, Q40A, Q41A, + 11 demographics
    anx_feat  = np.array([[
        a.Q2A, a.Q4A, a.Q7A, a.Q9A, a.Q15A, a.Q19A, a.Q20A,
        a.Q23A, a.Q25A, a.Q28A, a.Q30A, a.Q36A, a.Q40A, a.Q41A,
        *dv
    ]])
    anx_scaled = anx_scaler.transform(anx_feat)
    anx_cls    = int(anxiety_model.predict(anx_scaled)[0])
    anx_proba  = anxiety_model.predict_proba(anx_scaled)[0]

    # ── Stress Model ────────────────────────────────────────────────
    # Feature order: Q1A, Q6A, Q8A, Q11A, Q12A, Q14A, Q18A, Q22A,
    #                Q27A, Q29A, Q32A, Q33A, Q35A, Q39A, + 11 demographics
    str_feat  = np.array([[
        s.Q1A, s.Q6A, s.Q8A, s.Q11A, s.Q12A, s.Q14A, s.Q18A,
        s.Q22A, s.Q27A, s.Q29A, s.Q32A, s.Q33A, s.Q35A, s.Q39A,
        *dv
    ]])
    str_scaled = str_scaler.transform(str_feat)
    str_cls    = int(stress_model.predict(str_scaled)[0])
    str_proba  = stress_model.predict_proba(str_scaled)[0]

    # Helper: convert probability array → readable dict
    def pdict(proba):
        return {SEVERITY_LABELS[i]: round(float(p), 4) for i, p in enumerate(proba)}

    return {
        "assessment": {
            "depression": {
                "severity_class"      : dep_cls,
                "severity_label"      : SEVERITY_LABELS[dep_cls],
                "severity_color"      : SEVERITY_COLORS[dep_cls],
                "confidence"          : round(float(dep_proba.max()), 4),
                "probabilities"       : pdict(dep_proba),
                "wellness_tip"        : WELLNESS_TIPS["depression"][dep_cls],
                "requires_urgent_care": dep_cls >= 3
            },
            "anxiety": {
                "severity_class"      : anx_cls,
                "severity_label"      : SEVERITY_LABELS[anx_cls],
                "severity_color"      : SEVERITY_COLORS[anx_cls],
                "confidence"          : round(float(anx_proba.max()), 4),
                "probabilities"       : pdict(anx_proba),
                "wellness_tip"        : WELLNESS_TIPS["anxiety"][anx_cls],
                "requires_urgent_care": anx_cls >= 3
            },
            "stress": {
                "severity_class"      : str_cls,
                "severity_label"      : SEVERITY_LABELS[str_cls],
                "severity_color"      : SEVERITY_COLORS[str_cls],
                "confidence"          : round(float(str_proba.max()), 4),
                "probabilities"       : pdict(str_proba),
                "wellness_tip"        : WELLNESS_TIPS["stress"][str_cls],
                "requires_urgent_care": str_cls >= 3
            }
        },
        "clinical_urgency"   : get_urgency_flag(dep_cls, anx_cls, str_cls),
        "overall_risk_score" : round((dep_cls + anx_cls + str_cls) / 12 * 100, 1),
        "disclaimer"         : (
            "This assessment is for screening purposes only and does not "
            "constitute a clinical diagnosis. Results should be reviewed by "
            "a qualified mental health professional."
        )
    }


# ══════════════════════════════════════════════════════════════════
#  SECTION 6 — API ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get("/", tags=["General"])
def home():
    """Welcome page — confirms the API is live and lists all endpoints."""
    return {
        "service"   : "DASS-42 Mental Health Assessment API",
        "status"    : "🟢 Online",
        "built_for" : "Telehealth & Wellness Platforms",
        "endpoints" : {
            "full_assessment"   : "POST /assess              — All 3 conditions in one call",
            "depression_screen" : "POST /screen/depression   — Depression only (14 questions)",
            "anxiety_screen"    : "POST /screen/anxiety      — Anxiety only (14 questions)",
            "stress_screen"     : "POST /screen/stress       — Stress only (14 questions)",
            "question_list"     : "GET  /reference/questions — All 42 questions for building your UI",
            "severity_guide"    : "GET  /reference/severity-scale",
            "health_check"      : "GET  /health",
            "interactive_docs"  : "GET  /docs"
        }
    }


@app.get("/health", tags=["General"])
def health():
    """
    Check if all 3 models are loaded and ready to accept requests.
    Use this endpoint from your platform's uptime monitoring dashboard.
    """
    if bundle is None:
        return JSONResponse(status_code=503, content={
            "status": "🔴 Unhealthy",
            "reason": "Model file not loaded.",
            "fix"   : "Place final_dass_system.pkl next to main.py and restart the server."
        })
    return {
        "status"            : "🟢 Healthy",
        "models_loaded"     : ["depression_model", "anxiety_model", "stress_model"],
        "scalers_loaded"    : ["dep_scaler", "anx_scaler", "str_scaler"],
        "features_per_model": 25,
        "severity_classes"  : list(SEVERITY_LABELS.values()),
        "timestamp"         : datetime.utcnow().isoformat() + "Z"
    }


@app.post("/assess", tags=["Assessment"])
def full_assessment(request: DASSRequest):
    """
    ## 🧠 Full DASS-42 Assessment — Main Endpoint

    **The primary endpoint for your telehealth or wellness app.**

    Send all 42 question answers + demographic info and receive:
    - Depression, Anxiety, and Stress **severity levels**
    - **Confidence scores** and probability breakdown per severity level
    - **Tailored wellness recommendations** per condition
    - **Clinical urgency flag** for the provider dashboard (ROUTINE / MODERATE / HIGH / URGENT)
    - **Overall risk score** from 0 to 100

    **Typical use:** After a user finishes the DASS-42 questionnaire in your app,
    POST their answers here and display the results on their results screen.
    """
    start  = time.time()
    result = run_prediction(request)

    return {
        **result,
        "user_id"      : request.user_id,
        "session_id"   : request.session_id,
        "assessed_at"  : datetime.utcnow().isoformat() + "Z",
        "processing_ms": round((time.time() - start) * 1000, 2)
    }


@app.post("/screen/depression", tags=["Quick Screen"])
def screen_depression(questions: DepressionQuestions, demographics: DemographicInfo):
    """
    ## 😔 Depression-Only Screening

    Send only the 14 depression questions + demographics.
    Returns severity + wellness tip in milliseconds.

    **Use case:** Pre-consultation mood screen, onboarding intake,
    or PHQ-style depression check within your app flow.
    """
    start = time.time()
    dv    = [demographics.education, demographics.urban, demographics.gender,
             demographics.engnat, demographics.age, demographics.screensize,
             demographics.religion, demographics.orientation, demographics.race,
             demographics.married, demographics.familysize]
    q     = questions

    feat   = np.array([[q.Q3A, q.Q5A, q.Q10A, q.Q13A, q.Q16A, q.Q17A, q.Q21A,
                         q.Q24A, q.Q26A, q.Q31A, q.Q34A, q.Q37A, q.Q38A, q.Q42A, *dv]])
    scaled = dep_scaler.transform(feat)
    cls    = int(depression_model.predict(scaled)[0])
    proba  = depression_model.predict_proba(scaled)[0]

    return {
        "domain"              : "Depression",
        "severity_class"      : cls,
        "severity_label"      : SEVERITY_LABELS[cls],
        "severity_color"      : SEVERITY_COLORS[cls],
        "confidence"          : round(float(proba.max()), 4),
        "probabilities"       : {SEVERITY_LABELS[i]: round(float(p), 4) for i, p in enumerate(proba)},
        "wellness_tip"        : WELLNESS_TIPS["depression"][cls],
        "requires_urgent_care": cls >= 3,
        "processing_ms"       : round((time.time() - start) * 1000, 2),
        "disclaimer"          : "Screening result only — not a clinical diagnosis."
    }


@app.post("/screen/anxiety", tags=["Quick Screen"])
def screen_anxiety(questions: AnxietyQuestions, demographics: DemographicInfo):
    """
    ## 😰 Anxiety-Only Screening

    Send only the 14 anxiety questions + demographics.

    **Use case:** Pre-therapy check-in, panic disorder monitoring,
    or generalised anxiety tracking within your wellness platform.
    """
    start = time.time()
    dv    = [demographics.education, demographics.urban, demographics.gender,
             demographics.engnat, demographics.age, demographics.screensize,
             demographics.religion, demographics.orientation, demographics.race,
             demographics.married, demographics.familysize]
    q     = questions

    feat   = np.array([[q.Q2A, q.Q4A, q.Q7A, q.Q9A, q.Q15A, q.Q19A, q.Q20A,
                         q.Q23A, q.Q25A, q.Q28A, q.Q30A, q.Q36A, q.Q40A, q.Q41A, *dv]])
    scaled = anx_scaler.transform(feat)
    cls    = int(anxiety_model.predict(scaled)[0])
    proba  = anxiety_model.predict_proba(scaled)[0]

    return {
        "domain"              : "Anxiety",
        "severity_class"      : cls,
        "severity_label"      : SEVERITY_LABELS[cls],
        "severity_color"      : SEVERITY_COLORS[cls],
        "confidence"          : round(float(proba.max()), 4),
        "probabilities"       : {SEVERITY_LABELS[i]: round(float(p), 4) for i, p in enumerate(proba)},
        "wellness_tip"        : WELLNESS_TIPS["anxiety"][cls],
        "requires_urgent_care": cls >= 3,
        "processing_ms"       : round((time.time() - start) * 1000, 2),
        "disclaimer"          : "Screening result only — not a clinical diagnosis."
    }


@app.post("/screen/stress", tags=["Quick Screen"])
def screen_stress(questions: StressQuestions, demographics: DemographicInfo):
    """
    ## 😤 Stress-Only Screening

    Send only the 14 stress questions + demographics.

    **Use case:** Workplace wellness apps, weekly burnout check-ins,
    or employee mental health programmes.
    """
    start = time.time()
    dv    = [demographics.education, demographics.urban, demographics.gender,
             demographics.engnat, demographics.age, demographics.screensize,
             demographics.religion, demographics.orientation, demographics.race,
             demographics.married, demographics.familysize]
    q     = questions

    feat   = np.array([[q.Q1A, q.Q6A, q.Q8A, q.Q11A, q.Q12A, q.Q14A, q.Q18A,
                         q.Q22A, q.Q27A, q.Q29A, q.Q32A, q.Q33A, q.Q35A, q.Q39A, *dv]])
    scaled = str_scaler.transform(feat)
    cls    = int(stress_model.predict(scaled)[0])
    proba  = stress_model.predict_proba(scaled)[0]

    return {
        "domain"              : "Stress",
        "severity_class"      : cls,
        "severity_label"      : SEVERITY_LABELS[cls],
        "severity_color"      : SEVERITY_COLORS[cls],
        "confidence"          : round(float(proba.max()), 4),
        "probabilities"       : {SEVERITY_LABELS[i]: round(float(p), 4) for i, p in enumerate(proba)},
        "wellness_tip"        : WELLNESS_TIPS["stress"][cls],
        "requires_urgent_care": cls >= 3,
        "processing_ms"       : round((time.time() - start) * 1000, 2),
        "disclaimer"          : "Screening result only — not a clinical diagnosis."
    }


@app.get("/reference/severity-scale", tags=["Reference"])
def severity_reference():
    """
    Returns the DASS-42 official 5-level severity classification.
    Use this to build colour-coded badges, legends, or tooltips in your UI.
    """
    return {
        "scale": [
            {"class": 0, "label": "Normal",          "color": "green",    "description": "Within healthy range. No clinical concern."},
            {"class": 1, "label": "Mild",             "color": "yellow",   "description": "Some symptoms present. Monitor and apply self-care."},
            {"class": 2, "label": "Moderate",         "color": "orange",   "description": "Noticeable symptoms. Professional check-in recommended."},
            {"class": 3, "label": "Severe",           "color": "red",      "description": "Significant symptoms. Prompt professional evaluation advised."},
            {"class": 4, "label": "Extremely Severe", "color": "dark_red", "description": "Critical symptoms. Urgent professional support strongly recommended."}
        ],
        "answer_scale": {
            "0": "Did not apply to me at all",
            "1": "Applied to me to some degree / some of the time",
            "2": "Applied to me to a considerable degree / good part of the time",
            "3": "Applied to me very much / most of the time"
        }
    }


@app.get("/reference/questions", tags=["Reference"])
def question_reference():
    """
    Returns all 42 DASS questions grouped by domain.
    Use this to dynamically render the questionnaire inside your app — no hardcoding needed!
    """
    return {
        "instructions" : "Rate each item based on how much it applied to you OVER THE PAST WEEK.",
        "answer_scale" : {"0": "Never", "1": "Sometimes", "2": "Often", "3": "Almost always"},
        "depression_questions": {
            "Q3A" : "I couldn't seem to experience any positive feeling at all",
            "Q5A" : "I just couldn't seem to get going",
            "Q10A": "I felt that I had nothing to look forward to",
            "Q13A": "I felt sad and depressed",
            "Q16A": "I felt that I had lost interest in just about everything",
            "Q17A": "I felt I wasn't worth much as a person",
            "Q21A": "I felt that life was meaningless",
            "Q24A": "I couldn't seem to get any enjoyment out of the things I did",
            "Q26A": "I felt down-hearted and blue",
            "Q31A": "I was unable to become enthusiastic about anything",
            "Q34A": "I felt I was pretty worthless",
            "Q37A": "I felt that life wasn't worthwhile",
            "Q38A": "I could see nothing in the future to be hopeful about",
            "Q42A": "I found it difficult to work up the initiative to do things"
        },
        "anxiety_questions": {
            "Q2A" : "I was aware of dryness of my mouth",
            "Q4A" : "I experienced breathing difficulty",
            "Q7A" : "I experienced trembling (e.g., in the hands)",
            "Q9A" : "I was worried about situations in which I might panic",
            "Q15A": "I felt I was close to panic",
            "Q19A": "I was aware of the action of my heart in the absence of physical exertion",
            "Q20A": "I felt scared without any good reason",
            "Q23A": "I had difficulty swallowing",
            "Q25A": "I was aware of changes in my heart rate",
            "Q28A": "I felt I was about to have a panic attack",
            "Q30A": "I felt terror",
            "Q36A": "I was scared without any good reason",
            "Q40A": "I felt my heart beating fast",
            "Q41A": "I was afraid that I would be thrown by some trivial but unfamiliar task"
        },
        "stress_questions": {
            "Q1A" : "I found it hard to wind down",
            "Q6A" : "I tended to over-react to situations",
            "Q8A" : "I felt that I was using a lot of nervous energy",
            "Q11A": "I found myself getting agitated",
            "Q12A": "I found it difficult to relax",
            "Q14A": "I was intolerant of anything that kept me from getting on with what I was doing",
            "Q18A": "I felt that I was rather touchy",
            "Q22A": "I found it hard to calm down after something upset me",
            "Q27A": "I was irritable",
            "Q29A": "I found it hard to calm down",
            "Q32A": "I found it difficult to tolerate interruptions to what I was doing",
            "Q33A": "I was in a state of nervous tension",
            "Q35A": "I felt that demands were building so high that I could not overcome them",
            "Q39A": "I became irritated when things didn't go my way"
        }
    }


# ══════════════════════════════════════════════════════