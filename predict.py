import pickle

import uvicorn
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel, Field


class Client(BaseModel):
    RhythmScore: float = Field(..., ge=-100.0)
    AudioLoudness: float = Field(..., ge=-100.0)
    VocalContent: float = Field(..., ge=-100.0)
    AcousticQuality: float = Field(..., ge=-100.0)
    InstrumentalScore: float = Field(..., ge=-100.0)
    LivePerformanceLikelihood: float = Field(..., ge=-100.0)
    MoodScore: float = Field(..., ge=-100.0)
    TrackDurationMs: float = Field(..., ge=-100.0)
    Energy: float = Field(..., ge=-100.0)


class PredictResponse(BaseModel):
    BeatsPerMinute: float


app = FastAPI(title="Beats-Per-Minute in Song")

# Load DictVectorizer and Model
with open('xgboost_model0.1_4_1.8995562787677835_0.8277050661510729_0.6877153806619976_9.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


def predict_single(client: Client):
    X = dv.transform([client.dict()])
    features = list(dv.get_feature_names_out())

    # Create a DMatrix for prediction
    dmatrix = xgb.DMatrix(X, feature_names=features)

    # Predict days in shelter
    y_pred = model.predict(dmatrix)[0]

    result = {
        'Beats in song': float(y_pred)
    }
    return float(y_pred)


@app.post("/predict")
def predict(client: Client) -> PredictResponse:
    prob = predict_single(client)
    return PredictResponse(BeatsPerMinute=prob)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
