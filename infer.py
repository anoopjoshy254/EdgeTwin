import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf


def main():
    models_dir = "models"
    preprocessor_path    = os.path.join(models_dir, "preprocessor.pkl")
    model_path           = os.path.join(models_dir, "model.h5")
    cat_values_path      = os.path.join(models_dir, "categorical_values.pkl")

    if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
        print(f"Error: Missing files. Ensure {model_path} and {preprocessor_path} exist. Run train.py first.")
        return

    # Load artifacts
    print("[*] Loading trained preprocessor and model...")
    preprocessor  = joblib.load(preprocessor_path)
    feature_names = joblib.load(os.path.join(models_dir, "feature_names.pkl"))
    model         = tf.keras.models.load_model(model_path)

    # Load valid category values saved during training
    cat_valid_values = {}
    if os.path.exists(cat_values_path):
        cat_valid_values = joblib.load(cat_values_path)

    print("--------------------------------------------------")
    print("Machine Failure Detection Interactive Inference")
    print("--------------------------------------------------")

    # Extract expected column names directly from the preprocessor
    try:
        num_cols = list(preprocessor.transformers_[0][2])
        cat_cols = list(preprocessor.transformers_[1][2])
    except Exception as e:
        print("Warning: Could not extract feature columns from preprocessor.")
        print(e)
        return

    inputs = {}

    # ---- Numerical inputs ----
    print("\nPlease enter numerical sensor values:")
    for col in num_cols:
        while True:
            try:
                val = input(f"  {col}: ")
                inputs[col] = [float(val)]
                break
            except ValueError:
                print("    Invalid input. Please enter a number.")

    # ---- Categorical inputs ----
    if cat_cols:
        print("\nPlease enter categorical values:")
        for col in cat_cols:
            # Show the actual valid options from the training dataset
            if col in cat_valid_values and cat_valid_values[col]:
                options = " / ".join(str(v) for v in cat_valid_values[col])
                prompt = f"  {col} (options: {options}): "
            else:
                prompt = f"  {col}: "

            while True:
                val = input(prompt).strip()
                # Validate against known values if available
                if col in cat_valid_values and cat_valid_values[col]:
                    valid = [str(v) for v in cat_valid_values[col]]
                    if val in valid:
                        inputs[col] = [val]
                        break
                    else:
                        print(f"    Invalid value '{val}'. Please choose from: {', '.join(valid)}")
                else:
                    inputs[col] = [val]
                    break

    # ---- Process & predict ----
    input_df = pd.DataFrame(inputs)
    print("\n[*] Processing inputs...")
    processed_input = preprocessor.transform(input_df)

    prediction_prob = model.predict(processed_input, verbose=0)[0][0]

    print("\n===============================")
    if prediction_prob > 0.5:
        print(" Prediction: FAILURE 🚨")
        print(f" Confidence: {prediction_prob * 100:.1f}%")
    else:
        print(" Prediction: NORMAL ✅")
        print(f" Confidence: {(1 - prediction_prob) * 100:.1f}%")
    print("===============================\n")


if __name__ == "__main__":
    main()
