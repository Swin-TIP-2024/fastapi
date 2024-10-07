from fastapi import FastAPI, File, UploadFile
import subprocess
from typing import Annotated
import uvicorn

from utils.extract_features import extract_binary_features
from utils.predict import predict_xgboost, predict_rf, predict_gan_rf

app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Hello World"}


@app.post("/predict/{model}")
async def decompile_apk(model, apk_file: Annotated[UploadFile, File()]):
    # Save the uploaded file
    file_location = f"assets/{apk_file.filename}"
    with open(file_location, "wb") as file:
        file.write(await apk_file.read())

    print("Running script")
    subprocess.call(f"assets/decompile_apk.sh {file_location}", shell=True)
    print("File decompiled")

    print("Extracting features")
    binary_feature_vector = extract_binary_features()
    print("Features extracted in binary format")

    print("Predicting using models")
    if model == "xgboost":
        prediction = predict_xgboost(binary_feature_vector)
    elif model == "km-rf":
        prediction = predict_rf(binary_feature_vector)
    elif model == "gan-rf":
        prediction = predict_gan_rf(binary_feature_vector)

    result = prediction["prediction"]
    probability = prediction["probability"]
    proba = probability[result]

    return {"prediction": int(result), "probability": float(proba)}


# Run Uvicorn server on port 8000
# uvicorn.run(app, host="0.0.0.0", port=8000)
