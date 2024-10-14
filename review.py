from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__,template_folder="templates")

# Load the trained model
model = load_model('model.h5')

# Function to preprocess the review text
def preprocess_review(review_text):
    # Tokenize the text
    words = review_text.lower().split()
    # Map words to indices based on the IMDb dataset
    word_to_index = imdb.get_word_index()
    review_indices = [word_to_index[word] + 3 for word in words if word in word_to_index]
    # Pad sequences to ensure uniform length
    review_padded = pad_sequences([review_indices], maxlen=100)
    return review_padded

# Function to predict sentiment
def predict_sentiment(review_text):
    review_padded = preprocess_review(review_text)
    # Predict sentiment (0 for negative, 1 for positive)
    prediction = model.predict(review_padded)[0]
    return "Positive" if prediction >= 0.5 else "Negative"

# Route to handle review submission
@app.route('/submit-review', methods=['POST'])
def submit_review():
    data = request.json
    review_text = data['review']
    sentiment = predict_sentiment(review_text)
    return jsonify({'sentiment': sentiment})

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/submit-review', methods=['POST'])
# def submit_review():
#     try:
#         data = request.json
#         review_text = data.get('review', '')
#         if not review_text:
#             return jsonify({'error': 'No review text provided'}), 400
        
#         sentiment = predict_sentiment(review_text)
#         return jsonify({'sentiment': sentiment})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
