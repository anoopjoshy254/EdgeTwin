import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore")

import time
import asyncio
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from google import genai
from google.genai import types as genai_types


class DigitalTwinSimulator:
    """
    Core simulation class used by the FastAPI server to stream sensor readings over WebSocket.
    """

    def __init__(self, csv_path: str, api_key: str = None):
        self.csv_path = csv_path
        self.api_key = api_key
        self.ai_enabled = False
        self.genai_client = None

        if api_key:
            try:
                self.genai_client = genai.Client(api_key=api_key)
                self.ai_enabled = True
                print(f"[AI] Gemini AI enabled (gemini-2.5-flash)")
            except Exception as e:
                print(f"[AI] Failed to initialise Gemini client: {e}")

        models_dir = os.path.join(os.path.dirname(__file__), "models")
        self.preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.pkl"))
        self.model = tf.keras.models.load_model(os.path.join(models_dir, "best_model.keras"))

        cat_path = os.path.join(models_dir, "categorical_values.pkl")
        self.cat_valid_values = joblib.load(cat_path) if os.path.exists(cat_path) else {}

        self.df = pd.read_csv(csv_path)
        self.machine_col = "Machine_Type"

    def get_machine_types(self):
        return list(self.cat_valid_values.get(self.machine_col, []))

    def _generate_ai_report_sync(self, machine_type, temp, vib, pwr, hum) -> str | None:
        """Synchronous Gemini call — run in a thread via asyncio.to_thread()."""
        if not self.ai_enabled or not self.genai_client:
            print("[AI] Skipping — AI not enabled.")
            return None
        try:
            print(f"[AI] Calling Gemini for {machine_type} (T={temp:.1f} V={vib:.1f}) ...")
            prompt = (
                f"You are an expert industrial AI assistant monitoring a {machine_type} machine.\n"
                f"A FAILURE anomaly was detected with these sensor readings:\n"
                f"- Temperature: {temp:.1f} °C\n"
                f"- Vibration: {vib:.1f} mm/s\n"
                f"- Power Usage: {pwr:.1f} kW\n"
                f"- Humidity: {hum:.1f}%\n\n"
                f"Write EXACTLY ONE short incident report (max 120 words) in plain text. "
                f"Do NOT write multiple reports, alternatives, or numbered options. "
                f"Structure it as:\n"
                f"FAILURE TYPE: [one sentence]\n"
                f"ROOT CAUSE: [one or two sentences based on the sensor values]\n"
                f"ACTION: [one sentence for the operator]"
            )
            response = self.genai_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=0.7)
            )
            text = response.text.strip()
            print(f"[AI] Report generated ({len(text)} chars)")
            return text
        except Exception as e:
            err_str = str(e)
            if 'RESOURCE_EXHAUSTED' in err_str or '429' in err_str or 'quota' in err_str.lower():
                print("[AI] Rate-limited — skipping this report, will retry next time.")
                return None  # Don't disable AI permanently — just skip this one
            print(f"[AI] Error: {err_str[:200]}")
            return None

    async def generate_ai_report(self, machine_type, temp, vib, pwr, hum) -> str | None:
        """Non-blocking async wrapper — runs Gemini in a thread pool."""
        return await asyncio.to_thread(
            self._generate_ai_report_sync, machine_type, temp, vib, pwr, hum
        )

    async def stream(self, machine_type: str, interval: float = 2.0):
        """
        Async generator: yields sensor + prediction dicts every `interval` seconds.
        AI report generated every 3 failures with 30s minimum cooldown.
        """
        machine_df = self.df[self.df[self.machine_col] == machine_type].copy()
        if machine_df.empty:
            print(f"[Stream] No data for machine type: {machine_type}")
            return

        shuffled = machine_df.sample(frac=1).reset_index(drop=True)
        idx = 0
        failure_count = 0
        last_report_time = 0.0
        COOLDOWN_SEC = 60  # Increased to 60s to avoid rate limiting

        print(f"[Stream] Starting stream for {machine_type} (AI={'ON' if self.ai_enabled else 'OFF'})")

        while True:
            if idx >= len(shuffled):
                shuffled = machine_df.sample(frac=1).reset_index(drop=True)
                idx = 0

            row = shuffled.iloc[idx]
            temp = float(row["Temperature"])
            vib  = float(row["Vibration"])
            pwr  = float(row["Power_Usage"])
            hum  = float(row.get("Humidity", 40.0))
            idx += 1

            input_data = pd.DataFrame({
                "Temperature": [temp],
                "Vibration":   [vib],
                "Power_Usage": [pwr],
                "Humidity":    [hum],
                "Machine_Type": [machine_type]
            })

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                processed = self.preprocessor.transform(input_data)

            prob = float(self.model.predict(processed, verbose=0)[0][0])
            is_failure = prob > 0.5
            conf = prob * 100 if is_failure else (1 - prob) * 100

            report = None
            if is_failure:
                failure_count += 1
                now = time.time()
                print(f"[Stream] FAILURE #{failure_count} (prob={prob:.3f})")
                if (failure_count % 3 == 0
                        and self.ai_enabled
                        and (now - last_report_time) >= COOLDOWN_SEC):
                    print(f"[Stream] Triggering AI report at failure #{failure_count}...")
                    # Non-blocking — doesn't stall the WebSocket
                    report = await self.generate_ai_report(machine_type, temp, vib, pwr, hum)
                    if report:
                        last_report_time = time.time()
                        print(f"[Stream] AI report ready — sending to client")

            payload = {
                "time":          datetime.now().strftime("%H:%M:%S"),
                "temp":          round(temp, 1),
                "vib":           round(vib, 1),
                "pwr":           round(pwr, 1),
                "hum":           round(hum, 1),
                "status":        "FAILURE" if is_failure else "NORMAL",
                "confidence":    round(conf, 1),
                "failure_count": failure_count,
                "report":        report
            }

            yield payload
            await asyncio.sleep(interval)
