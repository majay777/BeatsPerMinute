import numpy as np
import pickle
import streamlit as st

import xgboost as xgb


with open('xgboost_model0.1_4_1.8995562787677835_0.8277050661510729_0.6877153806619976_9.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# Creating a function for prediction
def beats_prediction(input_data):
    X = dv.transform([input_data])
    features = list(dv.get_feature_names_out())
    # Create a DMatrix for prediction
    dmatrix = xgb.DMatrix(X, feature_names=features)
    # Predict days in shelter
    y_pred = model.predict(dmatrix)[0]
    return float(y_pred)




def main():
    # Giving Title
    st.title('Beats Per Minute Prediction app')
    RhythmScore = st.text_input('RhythmScore of song track')
    AudioLoudness = st.text_input('AudioLoudness')
    VocalContent = st.text_input('VocalContent')
    AcousticQuality = st.text_input('AcousticQuality')
    InstrumentalScore = st.text_input('InstrumentalScore')
    LivePerformanceLikelihood = st.text_input('LivePerformanceLikelihood')
    MoodScore = st.text_input('MoodScore')
    TrackDurationMs = st.text_input('TrackDurationMs')
    Energy = st.text_input('Energy')

    data = {
        'RhythmScore': RhythmScore,
        'AudioLoudness': AudioLoudness,
        'VocalContent':VocalContent,
        'AcousticQuality':AcousticQuality,
        'InstrumentalScore': InstrumentalScore,
        'LivePerformanceLikelihood': LivePerformanceLikelihood,
        'MoodScore': MoodScore,
        'TrackDurationMs': TrackDurationMs,
        'Energy': Energy
    }
    # Code for prediction
    prediction = ''
    if st.button('Predict'):
        prediction = beats_prediction(data)

    st.success(prediction)

if __name__ == '__main__':
    main()