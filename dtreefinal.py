import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.models import Model, load_model
from keras.layers import Input
import joblib

model_path = 'best_transformer_model.h5'
transformer_model = load_model(model_path)

print("Model Layers:")
for layer in transformer_model.layers:
    print(layer.name)

layer_name = 'dense_11'

input_ids = Input(shape=(50,), dtype=tf.int32, name='input_ids')
attention_mask = Input(shape=(50,), dtype=tf.int32, name='attention_mask')

transformer_model = TFAutoModel.from_pretrained('distilbert-base-uncased')
transformer_outputs = transformer_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0, :]
feature_extractor = Model(inputs=[input_ids, attention_mask], outputs=transformer_outputs)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def extract_features(texts):
    max_length = 50
    tokenized_data = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors="tf")
    inputs = {
        'input_ids': tokenized_data['input_ids'], 
        'attention_mask': tokenized_data['attention_mask']
    }
    features = feature_extractor.predict([inputs['input_ids'], inputs['attention_mask']])
    return features

data_path = 'new model/data.csv'
data = pd.read_csv(data_path)
data = data.sample(frac=0.1, random_state=42)
passwords = data['password'].astype(str).tolist()
strengths = data['strength'].values

features = extract_features(passwords)

X_train, X_test, y_train, y_test = train_test_split(features, strengths, test_size=0.2, random_state=42)

random_forest = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

joblib.dump(random_forest, 'random_forest_model.pkl')
