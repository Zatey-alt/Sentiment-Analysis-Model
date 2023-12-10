from flask import Flask, request, jsonify
import joblib
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the saved model
model = joblib.load('sentiment_model.pkl')
# vectorizer = CountVectorizer()
vectorizer = joblib.load('vectorizer.pkl') 

# create a route that manages user request and does sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data['text']
        
        vectorized_text = vectorizer.transform([text])
        
        prediction = int(model.predict(vectorized_text)[0])
        return jsonify({'sentiment': prediction})
    
    except Exception as e:
        print('Exception in predict method, Something went wrong!: ', e)
        return jsonify({'Error': str(e)})
