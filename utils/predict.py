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


def predict_gan_rf(binary_feature_vector):
    # Reshape the feature vector for prediction
    binary_feature_vector = np.array(binary_feature_vector).reshape(1, -1)

    print("Loading GAN")
    gan_discriminator = load_model("models/Prototype5/discriminator_model.h5")
    gan_discriminator.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    print("Loading GAN")
    print("Loading RF model")
    rf_model = joblib.load("models/Prototype5/random_forest_model.pkl")
    print("Loading RF complete")

    # Set up thresholds
    base_threshold = 0.5
    gan_threshold = 0.5

    print("Making prediction")
    # Get GAN anomaly score
    anomaly_score = gan_discriminator(binary_feature_vector)

    # Make prediction using Random Forest
    prediction = rf_model.predict(binary_feature_vector)
    prediction_proba = rf_model.predict_proba(binary_feature_vector)

    # Adjust threshold based on GAN anomaly score
    if anomaly_score < 0.5:  # If GAN detects anomaly
        adjusted_threshold = base_threshold - 0.3  # Lower the threshold
    else:
        adjusted_threshold = base_threshold  # Keep the base threshold

    # Output the result based on both Random Forest and GAN
    if prediction_proba[0][1] >= adjusted_threshold and anomaly_score < gan_threshold:
        # Both Random Forest and GAN indicate malicious behavior
        prediction = 1
        print("The APK is predicted to be malicious by both Random Forest and GAN.")
        proba = prediction_proba[0][1] + 0.5
        print(f"Adjusted proba = {prediction_proba[0][1] + 0.5}")
    elif prediction[0] == 0 and anomaly_score >= gan_threshold:
        # Both Random Forest and GAN indicate benign behavior
        prediction = 0
        print("The APK is predicted to be benign by both Random Forest and GAN.")
        proba = prediction_proba[0][0] + 0.5
        print(f"Adjusted proba = {prediction_proba[0][0] + 0.5}")
    else:
        # Conflicting results; requires further analysis
        prediction = "Uncertain"
        print("The models provide conflicting results. APK flagged for review.")

    return {"prediction": prediction, "probability": prediction_proba[0]}
