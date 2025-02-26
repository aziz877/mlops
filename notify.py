import smtplib
import sys
import os
from email.mime.text import MIMEText
import subprocess

# Configuration - Use environment variables for security
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "mohamedazizallaya06@gmail.com"
SENDER_PASSWORD = "wfmz fxbb zdzs hpmg"
RECEIVER_EMAIL = "mohamedaziz.allaya@esprit.tn"

def send_email(subject, body):
    try:
        # Create the email message with the task output as the body
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL

        # Send the email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())

        print(f"Email notification sent successfully: {subject}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def run_task_and_notify(command, subject):
    """Runs a shell command, captures output, and sends an email with the result."""
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        output = result.stdout + "\n" + result.stderr  # Capture both stdout and stderr
        send_email(subject, output)
    except Exception as e:
        send_email(subject, f"Failed to execute command: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python notify.py \"Subject\" \"Command to Run\"")
        sys.exit(1)

    subject = sys.argv[1]
    command = sys.argv[2]
    run_task_and_notify(command, subject)

