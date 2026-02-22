import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import time
import random
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import argparse
from google import genai
from google.genai import types as genai_types

def generate_ai_report(client, machine_type, temp, vib, pwr, hum):
    """Calls Gemini to generate a brief incident report based on sensor readings."""
    try:
        prompt = (
            f"You are an expert industrial AI assistant monitoring a {machine_type} machine. "
            f"A critical FAILURE anomaly has just been detected with these precise sensor readings:\n"
            f"- Temperature: {temp:.1f} °C\n"
            f"- Vibration: {vib:.1f} mm/s\n"
            f"- Power Usage: {pwr:.1f} kW\n"
            f"- Humidity: {hum:.1f}%\n\n"
            f"Provide a short, highly unique, and structured incident report in plain text. "
            f"Crucially, vary your tone, format, and specific hypotheses each time you generate a report so they do not look identical. "
            f"Be creative but realistic based on the exact sensor values provided.\n"
            f"Include:\n"
            f"1) Probable failure type\n"
            f"2) Root cause analysis based strictly on the anomalous sensor readings\n"
            f"3) Immediate recommended action for the operator on the floor."
        )
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=genai_types.GenerateContentConfig(temperature=0.9)
        )
        return response.text.strip()
        
    except Exception as e:
        return f"Could not generate AI report. Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Digital Twin Simulator using real dataset values")
    parser.add_argument("--csv", type=str, required=True, help="Path to the training CSV file to pull real data from")
    parser.add_argument("--api-key", type=str, help="Optional: Gemini API Key for AI Incident Reports. Can also use GEMINI_API_KEY env var.")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: Dataset '{args.csv}' not found.")
        sys.exit(1)

    # Setup Gemini API
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    ai_enabled = False
    genai_client = None
    
    if api_key:
        try:
            genai_client = genai.Client(api_key=api_key)
            ai_enabled = True
        except Exception as e:
            print(f"Warning: Failed to configure Gemini AI. Reports disabled. ({e})")
    else:
        print("Warning: No Gemini API Key provided. Run with --api-key or set GEMINI_API_KEY to enable AI Incident Reports.")

    models_dir = "models"
    preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
    model_path = os.path.join(models_dir, "best_model.keras")
    cat_values_path = os.path.join(models_dir, "categorical_values.pkl")

    if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
        print(f"Error: Missing files. Ensure {model_path} and {preprocessor_path} exist.")
        return

    print("[*] Loading Digital Twin Backend System...")
    
    preprocessor = joblib.load(preprocessor_path)
    model = tf.keras.models.load_model(model_path)
    
    cat_valid_values = {}
    if os.path.exists(cat_values_path):
        cat_valid_values = joblib.load(cat_values_path)

    print("-" * 60)
    print(" 🏭 Machine Failure Digital Twin Simulator")
    print("-" * 60)

    # 1. Choose Machine Type
    machine_col = "Machine_Type"
    chosen_machine = "Unknown"
    
    if machine_col in cat_valid_values and cat_valid_values[machine_col]:
        options = cat_valid_values[machine_col]
        print(f"\nSelect a Machine Type to Simulate ({', '.join(options)}):")
        while True:
            choice = input("> ").strip()
            if choice in options:
                chosen_machine = choice
                break
            print(f"Invalid choice. Please select from: {', '.join(options)}")
    else:
        chosen_machine = input("Enter Machine Type (e.g., Lathe): ").strip()

    print(f"[*] Loading real historical data for {chosen_machine} from dataset...")
    df = pd.read_csv(args.csv)

    # We no longer need to find the target column to split normals/failures,
    # as we are just going to organically stream the entire dataset for this machine.
    machine_df = df[df[machine_col] == chosen_machine].copy()
    
    if machine_df.empty:
        print(f"Error: No data found for {chosen_machine} in the dataset.")
        sys.exit(1)

    print(f"[*] Found {len(machine_df)} historical data points to replay natively.")
    print(f"\n[*] Initializing Digital Twin for: {chosen_machine}")
    print("[*] Starting real-time sensor stream playing back true historical data.")
    if ai_enabled:
        print("[*] 🤖 AI Incident Reports are ENABLED. Will trigger every 3 FAILURES.")
    print("Press Ctrl+C to stop.\n")
    loop_interval_sec = 2
    cycles = 0
    failure_count = 0

    try:
        print(f"{'Time':<10} | {'Temp':<6} | {'Vib':<5} | {'Pwr':<4} | {'Hum':<4} || {'STATUS':<15} | {'Conf'}")
        print("-" * 75)

        shuffled_data = machine_df.sample(frac=1).reset_index(drop=True)
        data_index = 0

        while True:
            cycles += 1
            current_time = datetime.now().strftime("%H:%M:%S")

            if data_index >= len(shuffled_data):
                shuffled_data = machine_df.sample(frac=1).reset_index(drop=True)
                data_index = 0

            row = shuffled_data.iloc[data_index]
            temp = row["Temperature"]
            vib  = row["Vibration"]
            pwr  = row["Power_Usage"]
            hum  = row["Humidity"] if "Humidity" in row else 40.0
            data_index += 1

            input_data = pd.DataFrame({
                "Temperature": [temp],
                "Vibration": [vib],
                "Power_Usage": [pwr],
                "Humidity": [hum],
                "Machine_Type": [chosen_machine]
            })

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                processed_input = preprocessor.transform(input_data)

            prediction_prob = model.predict(processed_input, verbose=0)[0][0]

            if prediction_prob > 0.5:
                status_txt = "🚨 FAILURE"
                status_color = "\033[91m" 
                conf = prediction_prob * 100
            else:
                status_txt = "✅ NORMAL"
                status_color = "\033[92m"
                conf = (1 - prediction_prob) * 100

            reset_color = "\033[0m"

            sensor_str = f"{current_time:<10} | {temp:6.1f} | {vib:5.1f} | {pwr:4.1f} | {hum:4.1f}"
            pred_str = f"{status_color}{status_txt:<15}{reset_color} | {conf:5.1f}%"
            
            print(f"{sensor_str} || {pred_str}")

            # --- AI Incident Report Logic ---
            if prediction_prob > 0.5:
                failure_count += 1
                print(f"{' ':<10} [Failure #{failure_count} detected — report triggers on every 3rd]")
                if failure_count % 3 == 0 and ai_enabled:
                    print("\n" + "="*75)
                    print(f"🤖 GENERATING AI INCIDENT REPORT (after {failure_count} failures)...")
                    print("="*75)
                    
                    report_text = generate_ai_report(genai_client, chosen_machine, temp, vib, pwr, hum)
                    
                    print(report_text)
                    print("="*75 + "\n")

            time.sleep(loop_interval_sec)

    except KeyboardInterrupt:
        print("\n\n[*] Digital Twin Simulation Stopped.")

if __name__ == "__main__":
    main()
