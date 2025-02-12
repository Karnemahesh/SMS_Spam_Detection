from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the model and vectorizer
with open('model(1).pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        message_vector = vectorizer.transform([message])
        prediction = model.predict(message_vector)

        label = 'Spam' if prediction[0] == 1 else 'Ham'
        return render_template('result.html', message=message, label=label)

if __name__ == '__main__':
    app.run(debug=True)
