"""Twilio SMS sender."""

import os
from twilio.rest import Client


def send_sms(to: str, body: str, from_: str | None = None) -> dict:
    """
    Send an SMS via Twilio.

    Reads TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_FROM_NUMBER
    from environment variables (or .env via python-dotenv).
    """
    account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    from_number = from_ or os.environ["TWILIO_FROM_NUMBER"]

    client = Client(account_sid, auth_token)
    message = client.messages.create(body=body, from_=from_number, to=to)
    return {"sid": message.sid, "status": message.status, "to": to}
