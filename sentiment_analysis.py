import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import plotly.express as px
import os

# ---------------------------------------
# PATH SETTINGS
# ---------------------------------------
DATA_IN = r"C:\Users\makbu\Desktop\Amazon-Sentiment-Analysis\dataset\Reviews.csv"
OUTPUT_DIR = "outputs_fast"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Loading dataset...")
df = pd.read_csv(DATA_IN)

# Keep only Text column
df = df.dropna(subset=["Text"])
print(f"Rows loaded: {len(df)}")

# ---------------------------------------
# SENTIMENT ANALYSIS (FAST, NO CLEANING)
# ---------------------------------------
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

print("Running sentiment analysis...")
df["Sentiment"] = df["Text"].apply(get_sentiment)

# ---------------------------------------
# EMOTION DETECTION
# ---------------------------------------
def get_emotion(text):
    try:
        emotion = NRCLex(str(text)).top_emotions
        if emotion:
            return emotion[0][0]  # Top emotion
        else:
            return "none"
    except:
        return "none"

print("Extracting emotions...")
df["Emotion"] = df["Text"].apply(get_emotion)

# ---------------------------------------
# CHART 1: SENTIMENT DISTRIBUTION
# ---------------------------------------
sentiment_counts = df["Sentiment"].value_counts()
fig = px.pie(values=sentiment_counts.values,
             names=sentiment_counts.index,
             title="Sentiment Distribution")
fig.write_image(f"{OUTPUT_DIR}/sentiment_pie.png")

# ---------------------------------------
# CHART 2: EMOTION DISTRIBUTION
# ---------------------------------------
emotion_counts = df["Emotion"].value_counts()
fig2 = px.bar(x=emotion_counts.index,
              y=emotion_counts.values,
              title="Emotion Distribution",
              labels={'x': 'Emotion', 'y': 'Count'})
fig2.write_image(f"{OUTPUT_DIR}/emotion_bar.png")

# ---------------------------------------
# WORDCLOUD (NO CLEANING)
# ---------------------------------------
print("Generating WordCloud...")
all_text = " ".join(df["Text"].astype(str))
wc = WordCloud(width=2000, height=1000, background_color="white").generate(all_text)

plt.figure(figsize=(15, 7))
plt.imshow(wc)
plt.axis("off")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/wordcloud.png")

# ---------------------------------------
# CONFUSION MATRIX (using Score → label)
# ---------------------------------------
print("Creating Confusion Matrix...")

# Convert Score (1–5) → Ground truth labels
def score_to_label(score):
    if score >= 4:
        return "Positive"
    elif score == 3:
        return "Neutral"
    else:
        return "Negative"

df["TrueLabel"] = df["Score"].apply(score_to_label)

cm = confusion_matrix(df["TrueLabel"], df["Sentiment"], labels=["Positive", "Neutral", "Negative"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Positive", "Neutral", "Negative"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix: True Score vs VADER Sentiment")
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")

# ---------------------------------------
# EXPORT CSV
# ---------------------------------------
df.to_csv(f"{OUTPUT_DIR}/Reviews_with_sentiment_emotion_fast.csv", index=False)

print("\nALL DONE! 🎉")
print(f"Outputs saved in folder: {OUTPUT_DIR}")
