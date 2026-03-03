# DASS-42 Mental Health Prediction API

A concise and robust FastAPI backend designed to process the 42 questions of the DASS-42 questionnaire along with 11 demographic features. It predicts the clinical severity classification of Depression, Anxiety, and Stress levels using an ensemble of saved machine-learning models.

## Features
- **Ultra-concise implementation**: Efficiently parses 42 individual items by dynamically routing subscale items to matching models via index mappings.
- **Robust against dimensionality mismatch**: Protects against unexpected user input shapes by parsing data exactly as the `.pkl` bundled models expect it (14 items + 11 demographic variables per model).
- **Easy deployment**: Simply start the server and push JSON payloads through standard REST methodology.

## Getting Started

### 1. Requirements
Ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
Run the API locally using `uvicorn`:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
*Note: Make sure your `final_dass_system.pkl` is located in the root repository alongside the scripts.*

## API Endpoints

### 1. Health Check
`GET /`
Returns the status of the API and verifies if the Pickled models are successfully loaded in memory.

### 2. Perform Prediction
`POST /predict`
Accepts a structured JSON payload containing the 42 DASS answers + 11 demographics.

**Request Body Example:**
```json
{
  "answers": [
    0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 3, 3, 2, 1, 0, 0,
    1, 2, 3, 3, 2, 1, 1, 2, 0, 0, 1, 2, 3, 1, 2, 3, 3, 2, 1, 0,
    1, 2
  ],
  "education": 3,
  "urban": 2,
  "gender": 1,
  "engnat": 1,
  "age": 25,
  "screensize": 15.6,
  "religion": 4,
  "orientation": 1,
  "race": 60,
  "married": 1,
  "familysize": 4
}
```

**Response Example:**
```json
{
  "depression_level": 2,
  "depression_severity": "Moderate",
  "anxiety_level": 1,
  "anxiety_severity": "Mild",
  "stress_level": 4,
  "stress_severity": "Extremely Severe"
}
```

## DASS Answer Scale (0 - 3)
* 0 = Did not apply to me at all
* 1 = Applied to me to some degree, or some of the time
* 2 = Applied to me to a considerable degree, or a good part of time
* 3 = Applied to me very much, or most of the time