import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from transformers import pipeline

df1 = pd.read_csv("high_popularity_spotify_data.csv")
df2 = pd.read_csv("low_popularity_spotify_data.csv")

df1["Level"] ="high"
df2["Level"] = "low"

df = pd.concat([d1, d2], ignore_index=True)

df = df.dropna(subset = ['valence', 'energy', 'track_name' 'track_artist'])

classifier = pipeline("zero-shot-classification", model = "facebook/bart-large-mnli")

russell_map = {
    'happy': (0.9, 0.75),
    'excited': (0.85, 0.95),
    'relaxed': (0.75, 0.3),
    'angry': (0.1, 0.9),
    'fearful': (0.2, 0.85),
    'sad': (0.2, 0.2),
    'depressed': (0.15, 0.1),
    'bored': (0.3, 0.15),
    'confused': (0.5, 0.6),
    'loved': (0.85, 0.5),
    'grateful': (0.8, 0.6),
    'neutral': (0.5, 0.5)
}

emotion = [key for key in russell_map]
userInput = input("How are you feeling?").strip().lower()

prediction = classifier(user_input, emotion)
predEmotion = prediction["labels"][0]
valArousal = russell_map[emotion]





X = df[['valence', 'energy']]
y = df.index


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X,y)



user_song = np.array(valArousal).reshape(-1,1)

songIndex = knn.predict(user_song)[0]
song_suggestion = df.iloc[songIndex]
             

print( "Here listen to this song based on how your feel!")
print(f" {song_suggestion['track_name']} by {song_suggestion['track_artist']}")
