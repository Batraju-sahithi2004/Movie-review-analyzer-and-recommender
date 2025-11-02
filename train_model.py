# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Download NLTK stopwords
nltk.download('stopwords')

# -------------------------------
# STEP 1: Load dataset
# -------------------------------
df = pd.read_csv("IMBD Dataset.csv")  # replace with your CSV file
reviews = df['review'].tolist()
  
# -------------------------------
# STEP 2: Preprocess text
# -------------------------------
stemmer = PorterStemmer()
stop_words = stopwords.words('english')

def preprocess_text(text):
    """Clean and stem text"""
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

df['cleaned'] = df['review'].apply(preprocess_text)

# -------------------------------
# STEP 3: Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned']).toarray()
y = df['sentiment']

# -------------------------------
# STEP 4: Split and train
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000, multi_class='ovr')
model.fit(X_train, y_train)

# -------------------------------
# STEP 5: Evaluate model
# -------------------------------
y_pred = model.predict(X_test)
print("✅ Classification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# STEP 6: Save model
# -------------------------------
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
print("💾 model.pkl and vectorizer.pkl saved successfully!")
