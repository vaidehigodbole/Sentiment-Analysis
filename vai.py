import streamlit as st
from textblob import TextBlob
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import cleantext
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("Reviews.csv")
review_text = df['Text']

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Compute sentiment scores and subjectivity
sentiment_scores = []
blob_subj = []
for review in review_text:
    sentiment_scores.append(analyzer.polarity_scores(review)["compound"])
    blob = TextBlob(review)
    blob_subj.append(blob.subjectivity)

# Categorize sentiment scores
def categorize_sentiment(score):
    if score > 0.8:
        return "Highly Positive"
    elif score > 0.4:
        return "Positive"
    elif -0.4 <= score <= 0.4:
        return "Neutral"
    elif score < -0.4:
        return "Negative"
    else:
        return "Highly Negative"

sentiment_classes = [categorize_sentiment(score) for score in sentiment_scores]

# Streamlit UI
st.title("Sentiment Analysis on Customer Feedback")

st.sidebar.header("User Input")
user_input = st.sidebar.text_area("Enter the Feedback:")

if user_input:
    blob = TextBlob(user_input)
    user_sentiment_score = analyzer.polarity_scores(user_input)['compound']
    user_sentiment_class = categorize_sentiment(user_sentiment_score)
    
    st.sidebar.write("**VADER Sentiment Class:**", user_sentiment_class)
    st.sidebar.write("**VADER Sentiment Score:**", user_sentiment_score)        
    st.sidebar.write("**TextBlob Polarity:**", blob.sentiment.polarity)
    st.sidebar.write("**TextBlob Subjectivity:**", blob.sentiment.subjectivity)

pre = st.sidebar.text_input('Clean Text: ')
if pre:
    cleaned_text = cleantext.clean(pre, clean_all=False, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True)
    st.sidebar.write("Cleaned Text:", cleaned_text)
else:
    st.sidebar.write("No Text has been provided for cleaning")

# Graphical Representation
st.subheader("Graphical Representation of Data")
plt.figure(figsize=(10,6))

sentiment_scores_by_class = {k: [] for k in set(sentiment_classes)}
for score, sentiment_class in zip(sentiment_scores, sentiment_classes):
    sentiment_scores_by_class[sentiment_class].append(score)

for sentiment_class, scores in sentiment_scores_by_class.items():
    plt.hist(scores, label=sentiment_class, alpha=0.5)

plt.xlabel("Sentiment score")
plt.ylabel("Count")
plt.title("Score Distribution by Class")
plt.legend()
st.pyplot(plt)

# Adding new columns to DataFrame
df["Sentiment Class"] = sentiment_classes
df["Sentiment Score"] = sentiment_scores
df["Subjectivity"] = blob_subj

# Selecting columns to display
new_df = df[["Score", "Text", "Sentiment Score", "Sentiment Class", "Subjectivity"]]
st.subheader("Input Dataframe")
st.dataframe(new_df.head(10), use_container_width=True)
