"""
Discord webhook notifications.
Supports any monitor type via config-driven title/color/emoji.
Parses LLM output into Discord embed fields when the model uses
the structured "**Label:** value" format.
"""

import re
import httpx
from datetime import datetime


# ── Field parsing ─────────────────────────────────────────────────────────────

_FIELD_RE = re.compile(r"^\*\*(.+?)\*\*[:\s]+(.+)$")


def _parse_fields(text: str) -> tuple[list[dict], str]:
    """
    Split LLM output into Discord embed fields + leftover prose.

    Lines matching "**Label:** value" become inline fields (up to 3 per row).
    Any remaining lines become the embed description.
    """
    fields = []
    leftover = []

    for line in text.strip().splitlines():
        m = _FIELD_RE.match(line.strip())
        if m:
            fields.append({
                "name": m.group(1).strip(),
                "value": m.group(2).strip(),
                "inline": True,
            })
        elif line.strip():
            leftover.append(line)

    # Discord allows max 25 fields; cap at 24 to be safe
    return fields[:24], "\n".join(leftover)


# ── Core sender ───────────────────────────────────────────────────────────────

def send_monitor_embed(
    summary: str,
    station_name: str,
    webhook_url: str,
    title: str = "Radio Update",
    color: int = 0x5865F2,
    mention: str = "",
) -> bool:
    """
    Send a structured Discord embed.

    The summary is split into embed fields (bold-labelled lines) plus a
    description block for any unlabelled prose. This gives a clean card layout
    rather than a wall of text.
    """
    if not webhook_url or webhook_url == "YOUR_DISCORD_WEBHOOK_URL_HERE":
        print(f"[Discord] Webhook not configured.\n--- Would have sent ---\n{title}\n{summary}\n---")
        return False

    fields, description = _parse_fields(summary)

    embed: dict = {
        "title": title,
        "color": color,
        "footer": {
            "text": f"{station_name}  •  {datetime.now().strftime('%b %d %Y  %H:%M')}",
        },
    }

    if description:
        embed["description"] = description
    if fields:
        embed["fields"] = fields

    # If nothing parsed into fields, just use the whole text as description
    if not fields and not description:
        embed["description"] = summary

    payload: dict = {"embeds": [embed]}
    if mention:
        payload["content"] = mention

    try:
        resp = httpx.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[Discord] Send failed: {e}")
        return False


# ── Convenience wrappers kept for backwards compatibility ─────────────────────

def weather_embed(summary: str, station_name: str, webhook_url: str, mention: str = "") -> bool:
    return send_monitor_embed(summary, station_name, webhook_url,
                              title="🌤️ Weather Report", color=0x3498DB, mention=mention)


def traffic_embed(summary: str, station_name: str, webhook_url: str, mention: str = "") -> bool:
    return send_monitor_embed(summary, station_name, webhook_url,
                              title="🚗 Traffic Alert", color=0xE74C3C, mention=mention)
