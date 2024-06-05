import joblib
from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(features, labels):
    print("Loading trained model...")
    model = joblib.load('models/random_forest_model.pkl')
    
    print("Making predictions on test set...")
    predictions = model.predict(features)
    accuracy = accuracy_score(labels, predictions)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print("Generating classification report...")
    report = classification_report(labels, predictions)
    print(report)
