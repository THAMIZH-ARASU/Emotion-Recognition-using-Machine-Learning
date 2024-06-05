from src.data_preprocessing import load_data, preprocess_data
from src.feature_extraction import extract_features
from src.train_model import train_random_forest  # Update import
from src.evaluate_model import evaluate_model
from src.utils import create_directories

def main():
    # Create necessary directories
    create_directories(['models', 'data/train', 'data/test'])
    
    # Load and preprocess training data
    train_images, train_labels = load_data('data/train')
    train_images = preprocess_data(train_images)
    train_features = extract_features(train_images)
    
    # Train the model - changed to Random Forest
    train_random_forest(train_features, train_labels)  # Changed function call
    
    # Load and preprocess test data
    test_images, test_labels = load_data('data/test')
    test_images = preprocess_data(test_images)
    test_features = extract_features(test_images)
    
    # Evaluate the model
    evaluate_model(test_features, test_labels)

if __name__ == "__main__":
    main()
