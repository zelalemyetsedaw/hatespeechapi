from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel
import tensorflow as tf



app = FastAPI()

class hatespeech(BaseModel):
    input_string: str
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pkl','rb') as f:
    loaded_le = pickle.load(f)
max_length = 100

with open('model_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)

# Load the model weights
loaded_model.load_weights('model_weights.weights.h5')

@app.get("/")
async def read_root():
    return {"Hello": "World"}
    
@app.post('/')
async def scoring_endpoint(data:hatespeech):
    input_sequence = tokenizer.texts_to_sequences([data.input_string])
    input_sequence_padded = tf.keras.preprocessing.sequence.pad_sequences(input_sequence, maxlen=max_length)

    # Use the loaded model to make a prediction
    predictions = loaded_model.predict(input_sequence_padded)
    predicted_label = loaded_le.inverse_transform(predictions.argmax(axis=1))

    return {"predicted_label": predicted_label[0]}

