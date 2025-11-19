# Beats Per Minute
___
![Song Beats](/images/79099.jpg?raw=true "Song Beats")

## Introduction
___

Dataset For this Project got from a Kaggle dataset. The Aim of this Project is to predict the beats of song per minute depending upon the various audio values of song.

## Column's of dataset
___
- RhythmScore- Quantifies the rhythmic complexity of the track
- AudioLoudnes-	Average loudness of the track in decibels
- VocalContent-	Proportion of vocals in the track
- AcousticQuality-	Score representing acoustic clarity and quality
- InstrumentalScore-	Score representing the strength of instrumental components
- LivePerformanceLikelihood-	Probability that the track could be performed live
- MoodScore-	Score representing overall mood of the track
- TrackDurationMs-	Duration of the track in milliseconds
- Energy-	Track energy level (high values = more energetic)
- BeatsPerMinute-	Target variable: BPM of the track

# EDA
___
This Dataset do conatin any duplicate/null values.

Distribution of Columns:

![Distribution](/images/distribution.jpg?raw=true "Distribution")

Histogram of columns: 
![Histogram](/images/histogram.jpg?raw=True "Histgram")

- BeatsPerMinute Distribution 
![Beats_Per_Minute](/images/beats_per_minute_dist.png?raw=True "Beats_Per_Minute")

- Correlation Matrix: 
![Correlation_Matrix](/images/Correlation_Matrix.jpg?raw=True "CorrelationMatrix")
___

# Steps To Reproduce

## Clone the repo

```
git clone https://github.com/majay777/BeatsPerMinute.git
```
## Run in Docker

```
docker build -t beatsperminute .
```
-To tun the predict app
```
docker run -it --rm -p 9696:9696 beatsperminute
```
___

## Run in Local Environment

Install uv
```
pip install uv
```
Initialize the Proejct
```commandline
uv init
```
Add required libraries
```commandline
uv add scikit-learn fastapi uvicorn pydantic streamlit xgboost pandas requests
```
Create the model file
```commandline
uv run python train.py
```
Run in one terminal
```commandline
uv run uvicorn predict:app --host 0.0.0.0 --port 9696 --reload
```

In Another Terminal
```commandline
uv run python requests.py
```
___
## Cloud Deployment

Deployed to Stramlit
Link- https://beatsperminute-y6.streamlit.app/

![Streamlit APP](/images/streamlit_app.png?raw=true "Streamlit App")