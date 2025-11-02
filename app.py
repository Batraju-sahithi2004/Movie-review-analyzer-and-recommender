import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')
# Keep negations
stop_words = [w for w in stop_words if w not in ['not', 'no', 'never']]

stemmer = PorterStemmer()

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Function to preprocess text
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

# Keyword-based override for strong negatives/positives
NEGATIVE_KEYWORDS = ["worst", "awful", "boring", "horrible", "terrible", "poor", "bad", "disappoint"]
POSITIVE_KEYWORDS = ["excellent", "amazing", "great", "fantastic", "best", "good", "love", "wonderful"]

def keyword_sentiment(review):
    review_lower = review.lower()
    if any(word in review_lower for word in NEGATIVE_KEYWORDS):
        return "Negative"
    elif any(word in review_lower for word in POSITIVE_KEYWORDS):
        return "Positive"
    else:
        return None  # let model decide

# Streamlit UI
st.title("Movie Review Sentiment & Recommendation")
st.write("Enter reviews of a movie (one per line):")

reviews_input = st.text_area("Reviews:", height=200)

if st.button("Analyze & Recommend"):
    if reviews_input.strip() == "":
        st.warning("Please enter some reviews!")
    else:
        reviews = reviews_input.strip().split("\n")
        reviews_clean = [preprocess_text(r) for r in reviews]
        reviews_vec = vectorizer.transform(reviews_clean)
        predictions_model = model.predict(reviews_vec)

        # Apply keyword override
        final_predictions = []
        for review, pred in zip(reviews, predictions_model):
            override = keyword_sentiment(review)
            final_predictions.append(override if override else pred)

        # Assign numeric scores
        sentiment_to_score = {"Positive": 5, "Neutral": 3, "Negative": 1}
        scores = [sentiment_to_score.get(p, 3) for p in final_predictions]
        avg_score = sum(scores) / len(scores)

        # Display each review's sentiment
        st.subheader("Individual Review Sentiments")
        for r, p in zip(reviews, final_predictions):
            st.write(f"**Review:** {r}")
            st.write(f"**Sentiment:** {p}")
            st.write("---")

        # Display recommendation
        st.subheader("Movie Recommendation Score (1–5)")
        st.write(f"**Average Score:** {avg_score:.1f} / 5")

        if avg_score >= 4:
            st.success("Recommendation: Highly Recommended ✅")
        elif avg_score >= 3:
            st.info("Recommendation: Maybe Watch 🤔")
        else:
            st.error("Recommendation: Not Recommended ❌")
