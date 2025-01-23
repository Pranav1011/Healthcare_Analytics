from src.predictive_modeling.predictor import prepare_data, train_model, evaluate_model, save_model
from pathlib import Path

def main():

    data_path = Path('data/processed/admissions.csv')  
    model_path = Path('src/models/ed_wait_time_model.pkl')  
    encoder_path = Path('src/models/encoder.pkl')          

    model_path.parent.mkdir(parents=True, exist_ok=True)

    print("Preparing data for modeling...")
    X_train, X_test, y_train, y_test, encoder = prepare_data(data_path)

    print("Training the model...")
    model = train_model(X_train, y_train)

    print("Evaluating the model...")
    mae, r2 = evaluate_model(model, X_test, y_test)
    print(f"Model evaluation completed. MAE: {mae:.2f}, R²: {r2:.2f}")

    print("Saving the model and encoder...")
    save_model(model, encoder, model_path, encoder_path)
    print("Model and encoder saved successfully!")

if __name__ == "__main__":
    main()