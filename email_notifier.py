"""
Email notification module for Digital Twin.
Sends failure alert emails via Gmail SMTP (App Password required).
"""
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


SENDER_EMAIL: str = ""
SENDER_PASSWORD: str = ""  # Gmail App Password


def configure(sender_email: str, sender_password: str):
    global SENDER_EMAIL, SENDER_PASSWORD
    SENDER_EMAIL = sender_email
    SENDER_PASSWORD = sender_password


def send_failure_alert(to_email: str, machine_type: str, report: str,
                       temp: float, vib: float, pwr: float, hum: float,
                       failure_count: int):
    """Send failure alert email in a background thread (non-blocking)."""
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("[Email] Sender not configured — skipping email.")
        return
    t = threading.Thread(
        target=_send,
        args=(to_email, machine_type, report, temp, vib, pwr, hum, failure_count),
        daemon=True
    )
    t.start()


def _send(to_email: str, machine_type: str, report: str,
          temp: float, vib: float, pwr: float, hum: float, failure_count: int):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── HTML email body ────────────────────────────────────────────────────
        html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #07101e; color: #cce0ff; margin: 0; padding: 0; }}
  .wrapper {{ max-width: 600px; margin: 0 auto; padding: 24px 16px; }}
  .header {{ background: linear-gradient(135deg, #0d1f35, #060f1e); border-left: 4px solid #ff2244;
             border-radius: 10px; padding: 20px 24px; margin-bottom: 20px; }}
  .header h1 {{ margin: 0 0 4px; font-size: 1.3rem; color: #ff4466; letter-spacing: 1px; }}
  .header .sub {{ color: #6a8aaa; font-size: 0.85rem; }}
  .sensors {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 16px 0; }}
  .sensor-chip {{ background: #0a1f35; border: 1px solid #1e3555; border-radius: 6px;
                  padding: 8px 14px; font-size: 0.82rem; color: #7aaace; }}
  .sensor-chip strong {{ color: #cce0ff; display: block; font-size: 1rem; }}
  .report-box {{ background: #0d1f35; border: 1px solid #ff224455; border-left: 3px solid #ff2244;
                 border-radius: 8px; padding: 16px 20px; white-space: pre-wrap;
                 font-size: 0.88rem; line-height: 1.6; color: #aabbd0; }}
  .footer {{ margin-top: 20px; font-size: 0.75rem; color: #3a5070; text-align: center; }}
  .badge {{ display: inline-block; background: #ff22440d; border: 1px solid #ff224488;
            color: #ff2244; font-size: 0.7rem; letter-spacing: 2px;
            padding: 4px 12px; border-radius: 20px; margin-bottom: 10px; }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <div class="badge">⚠ FAILURE ALERT #{failure_count}</div>
    <h1>🏭 {machine_type} Machine — Critical Failure Detected</h1>
    <div class="sub">Detected at {timestamp}</div>
  </div>

  <div class="sensors">
    <div class="sensor-chip"><strong>{temp:.1f} °C</strong>Temperature</div>
    <div class="sensor-chip"><strong>{vib:.1f} mm/s</strong>Vibration</div>
    <div class="sensor-chip"><strong>{pwr:.1f} kW</strong>Power</div>
    <div class="sensor-chip"><strong>{hum:.1f} %</strong>Humidity</div>
  </div>

  <p style="font-size:0.85rem;color:#6a8aaa;margin-bottom:8px;">🤖 <strong style="color:#cce0ff;">Gemini AI Incident Report</strong></p>
  <div class="report-box">{report}</div>

  <div class="footer">
    Machine Digital Twin · AI-Powered Industrial Monitoring<br>
    This is an automated alert. Do not reply to this email.
  </div>
</div>
</body>
</html>
"""

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🚨 [{machine_type}] Failure Alert #{failure_count} — Digital Twin"
        msg["From"]    = f"Digital Twin Alert <{SENDER_EMAIL}>"
        msg["To"]      = to_email

        msg.attach(MIMEText(report, "plain"))
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_email, msg.as_string())

        print(f"[Email] ✅ Alert sent to {to_email} (failure #{failure_count})")

    except Exception as e:
        print(f"[Email] ❌ Failed to send to {to_email}: {e}")
