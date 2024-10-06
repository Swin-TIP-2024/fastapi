import joblib
import numpy as np
from tensorflow.keras.models import load_model


def predict_xgboost(binary_feature_vector):
    # Reshape the feature vector for prediction (as encoder expects a 2D array)
    binary_feature_vector = np.array(binary_feature_vector).reshape(1, -1)

    print("Loading encoder")
    encoder = load_model("models/Prototype4/encoder_model.keras")
    print("Loading encoder complete")
    print("Loading XGBoost model")
    xgb_model = joblib.load("models/Prototype4/xgb_model.pkl")
    print("Loading XGBoost complete")
    print("Encoding features")
    # Transform the binary feature vector into latent features using the encoder
    latent_features = encoder.predict(binary_feature_vector)

    print("Making prediction")
    # Make predictions using the latent features
    prediction = xgb_model.predict(latent_features)
    prediction_proba = xgb_model.predict_proba(latent_features)

    return {"prediction": prediction[0], "probability": prediction_proba[0]}


def predict_rf(binary_feature_vector):
    # Reshape the feature vector for prediction
    binary_feature_vector = np.array(binary_feature_vector).reshape(1, -1)

    print("Loading KMeans model")
    kmeans_model = joblib.load("models/Prototype3/kmeans_model.pkl")
    print("Loading KMeans complete")
    print("Loading RF model")
    rf_model = joblib.load("models/Prototype3/random_forest_model.pkl")
    print("Loading RF complete")

    print("Making prediction")
    # Combine with KMeans cluster prediction
    kmeans_cluster = kmeans_model.predict(binary_feature_vector)
    feature_vector_with_cluster = np.hstack(
        (binary_feature_vector, kmeans_cluster.reshape(-1, 1))
    )

    # Make prediction using Random Forest
    prediction = rf_model.predict(feature_vector_with_cluster)
    prediction_proba = rf_model.predict_proba(feature_vector_with_cluster)

    return {"prediction": prediction[0], "probability": prediction_proba[0]}
