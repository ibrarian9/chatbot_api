import json
import pandas as pd
import re
import string
from flask import Flask, request, jsonify
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model

app = Flask(__name__)

# Load stopwords from file
stop_words = set()
with open('Dataset/stopword_id.txt', 'r', encoding='utf-8') as file:
    for line in file:
        stop_words.add(line.strip())

# Load dataset
with open('Dataset/dataset3.json') as f:
    data = json.load(f)

tags = []
inputs = []
responses = {}


# Function to remove stopwords
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])


for intent in data['intents']:
    responses[intent['tag']] = intent['responses']
    for line in intent['patterns']:
        # Preprocess text
        line = re.sub(r'[^\w\s.?]', '', line)

        # Remove stopwords
        line = remove_stopwords(line)

        inputs.append(line)
        tags.append(intent['tag'])

# Create a DataFrame
data_df = pd.DataFrame({"inputs": inputs, "tags": tags})

# Tokenizer
tokenizer = Tokenizer(num_words=2000, oov_token='<OOV>')
tokenizer.fit_on_texts(data_df['inputs'])

# Encode the outputs
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(data_df['tags'])
y_train = y_train.ravel()

# Padding sequences
sequences = tokenizer.texts_to_sequences(data_df['inputs'])
max_len = max(len(sequence) for sequence in sequences)
x_train = pad_sequences(sequences, maxlen=max_len, truncating='post', padding='post')  # Here, specify padding='post'

# Load the pre-trained model
model = load_model('Dataset/best_model.h5')


@app.route('/chat', methods=['POST'])
def chat():
    data_input = request.get_json()
    user_input = data_input['user_input']
    preprocessed_text = preprocess_input(user_input)

    try:
        input_seq = tokenizer.texts_to_sequences([preprocessed_text])
        input_seq = pad_sequences(input_seq, maxlen=max_len)
        predicted_label_seq = model.predict(input_seq)
        predicted_label = label_encoder.inverse_transform([predicted_label_seq.argmax(axis=-1)])[0]

        responses_for_tag = responses.get(predicted_label, ["Chatbot: Maaf, saya tidak memahami maksud Anda."])
        response = random.choice(responses_for_tag)

        return jsonify({'response': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def preprocess_input(user_input):
    user_input = user_input.lower()
    user_input = re.sub(r'[^\w\s.?]', '', user_input)
    user_input = user_input.translate(str.maketrans("", "", string.punctuation))
    words = user_input.split()
    words = [word for word in words if word not in stop_words]
    preprocessed_text = ' '.join(words)
    preprocessed_text = re.sub(' +', ' ', preprocessed_text)
    return preprocessed_text


if __name__ == '__main__':
    app.run(debug=True)
