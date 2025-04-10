import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Join tokens back into text
    return ' '.join(tokens)

# Load and preprocess the dataset
def load_dataset():
    # For demonstration, we'll create a small dummy dataset
    # In a real application, you would load your actual dataset
    data = {
        'text': [
            "Breaking: Scientists discover new breakthrough in cancer treatment",
            "SHOCKING: Aliens found living among us in secret! MUST READ!",
            "New study shows benefits of regular exercise",
            "You won't BELIEVE what this celebrity did to get rich quick!",
            "Local community opens new public library",
            "WARNING: Government hiding the truth about flat earth!",
            "Research indicates positive effects of meditation on mental health",
            "URGENT: Share this message or face bad luck for 7 years!",
            "New research confirms climate change impacts",
            "MIRACLE cure for all diseases found in common household item!",
        ],
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 for real, 0 for fake
    }
    return pd.DataFrame(data)

def train_model():
    # Load dataset
    df = load_dataset()
    
    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'],
        df['label'],
        test_size=0.2,
        random_state=42
    )
    
    # Create and fit TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train the model
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test_tfidf)
    
    # Print model performance
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and vectorizer
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

if __name__ == "__main__":
    train_model()
