from flask import Flask, render_template, request, jsonify
import pickle
import os
from model.train_model import preprocess_text

app = Flask(__name__)

# Load the trained model and vectorizer
def load_model():
    try:
        # Check if model files exist, if not, train the model
        if not (os.path.exists('model/model.pkl') and os.path.exists('model/vectorizer.pkl')):
            from model.train_model import train_model
            train_model()
        
        with open('model/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

model, vectorizer = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            # Get the news text from the form
            news_text = request.form['news_text']
            
            if not news_text:
                return render_template('result.html', 
                                    error="Please enter some text to analyze.")
            
            # Preprocess the text
            processed_text = preprocess_text(news_text)
            
            # Transform the text using vectorizer
            text_vector = vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = model.predict(text_vector)[0]
            probability = max(model.predict_proba(text_vector)[0])
            
            # Convert prediction to label and calculate confidence
            result = {
                'prediction': 'Real News' if prediction == 1 else 'Fake News',
                'confidence': round(probability * 100, 2),
                'original_text': news_text
            }
            
            return render_template('result.html', result=result)
            
    except Exception as e:
        return render_template('result.html', 
                             error=f"An error occurred: {str(e)}")

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)
