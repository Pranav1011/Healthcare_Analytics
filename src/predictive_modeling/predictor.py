import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def prepare_data(filepath):
    """Prepares data for predictive modeling."""
    data = pd.read_csv(filepath)
    features = ['admission_type', 'hour', 'admission_location', 'ethnicity']
    target = 'ed_wait_time'
    data = data.dropna(subset=features + [target])
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  
    encoded_features = encoder.fit_transform(data[features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(features))
    model_data = pd.concat([encoded_df, data[target].reset_index(drop=True)], axis=1)
    X = model_data.drop(columns=[target])
    y = model_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, encoder

def train_model(X_train, y_train):
    """Trains a Random Forest model and returns it."""
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model on the test data."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")
    return mae, r2


def save_model(model, encoder, model_path, encoder_path):
    """Saves the trained model and encoder."""
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    print(f"Model saved to {model_path}")
    print(f"Encoder saved to {encoder_path}")

def load_model(model_path, encoder_path):
    """Loads the trained model and encoder."""
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    print(f"Model loaded from {model_path}")
    print(f"Encoder loaded from {encoder_path}")
    return model, encoder