import requests

url = "http://localhost:9696/predict"
client = {
    "RhythmScore": 0.77394885,
    "AudioLoudness": -6.377050995,
    "VocalContent": 0.066823176,
    "AcousticQuality": 0.00000536,
    "InstrumentalScore": 0.063229518,
    "LivePerformanceLikelihood": 0.200380278,
    "MoodScore": 0.442237535,
    "TrackDurationMs": 194124.2433,
    "Energy": 0.205266667

}
response = requests.post(url, json=client)
print("Status:", response.status_code)
print("Headers:", response.headers)
print("Body:", response.text)
