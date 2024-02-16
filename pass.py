import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack, csr_matrix

data = pd.read_csv("data_Small.csv", on_bad_lines='skip')

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer()
    
    def fit(self, X, y=None):
        self.tfidf_vectorizer.fit(X['password'].astype(str))
        return self
    
    def transform(self, X):
        tfidf_features = self.tfidf_vectorizer.transform(X['password'].astype(str))
        length = np.array(X['password'].str.len()).reshape(-1, 1)
        length_sparse = csr_matrix(length)
        features = hstack([tfidf_features, length_sparse])
        return features 

pipeline = Pipeline([
    ('feature_extractor', FeatureExtractor()),
    ('svm', SVC(probability=True, random_state=42))
])

param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf'],
    'svm__gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(data[['password']], data['strength']) 

best_model = grid_search.best_estimator_

X_train, X_test, y_train, y_test = train_test_split(data[['password']], data['strength'], test_size=0.2, random_state=42)

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
#print(f"Model Accuracy: {accuracy:.3f}")

def predict_password_strength(password):
    prediction = best_model.predict(pd.DataFrame({'password': [password]}))
    strength_labels = {0: 'Weak', 1: 'Medium', 2: 'Strong'}
    print(f"Password: {password}, Predicted Strength: {strength_labels.get(prediction[0], 'Unknown')}")

if __name__ == "__main__":
    user_password = input("Enter a password to check its strength: ")
    predict_password_strength(user_password)
