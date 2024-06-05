from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

def train_random_forest(features, labels):
    print("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model on validation set...")
    predictions = model.predict(X_val)
    
    print("Saving trained model...")
    joblib.dump(model, 'models/random_forest_model.pkl')
