import os
import json
import argparse
from typing import List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import email_notifier

from digital_twin_core import DigitalTwinSimulator

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Digital Twin FastAPI Server")
parser.add_argument("--csv",              required=True,  help="Path to machine_failure_dataset.csv")
parser.add_argument("--api-key",          default=None,   help="Gemini API Key")
parser.add_argument("--sender-email",     default=None,   help="Gmail address to send alerts from")
parser.add_argument("--sender-password",  default=None,   help="Gmail App Password for sender account")
args, _ = parser.parse_known_args()

# ── Configure email notifier ──────────────────────────────────────────────────
if args.sender_email and args.sender_password:
    email_notifier.configure(args.sender_email, args.sender_password)
    print(f"[Email] Configured sender: {args.sender_email}")
else:
    print("[Email] No sender credentials provided — email alerts disabled. "
          "Use --sender-email and --sender-password to enable.")

# ── Singleton simulator ───────────────────────────────────────────────────────
simulator = DigitalTwinSimulator(csv_path=args.csv, api_key=args.api_key)

# ── In-memory state ───────────────────────────────────────────────────────────
registered_devices: List[dict] = []
registered_email: Optional[str] = None   # single registered alert email

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Digital Twin API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────
class Device(BaseModel):
    name: str
    machine_type: str
    serial_number: str
    location: str


class EmailRegistration(BaseModel):
    email: str


# ── REST endpoints ────────────────────────────────────────────────────────────
@app.get("/api/machines")
def get_machines():
    return {"machines": simulator.get_machine_types()}


@app.get("/api/devices")
def list_devices():
    return {"devices": registered_devices}


@app.post("/api/devices")
def register_device(device: Device):
    d = device.dict()
    d["id"] = len(registered_devices) + 1
    registered_devices.append(d)
    return {"success": True, "device": d}


@app.post("/api/register-email")
def register_email(body: EmailRegistration):
    global registered_email
    registered_email = body.email.strip()
    print(f"[Email] Registered alert email: {registered_email}")
    return {"success": True, "email": registered_email}


@app.get("/api/register-email")
def get_registered_email():
    return {"email": registered_email}


@app.delete("/api/register-email")
def unregister_email():
    global registered_email
    registered_email = None
    return {"success": True}


# ── WebSocket stream ──────────────────────────────────────────────────────────
@app.websocket("/ws/{machine_type}")
async def websocket_stream(websocket: WebSocket, machine_type: str):
    await websocket.accept()
    try:
        async for payload in simulator.stream(machine_type, interval=2.0):
            # If a report was just generated, email it to the registered user
            if payload.get("report") and registered_email:
                email_notifier.send_failure_alert(
                    to_email=registered_email,
                    machine_type=machine_type,
                    report=payload["report"],
                    temp=payload["temp"],
                    vib=payload["vib"],
                    pwr=payload["pwr"],
                    hum=payload["hum"],
                    failure_count=payload["failure_count"]
                )
            await websocket.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(json.dumps({"error": str(e)}))


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
