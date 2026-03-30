"""Discord webhook notifications."""

import httpx
from datetime import datetime


def send_discord(
    webhook_url: str,
    title: str,
    description: str,
    color: int | None = None,
    mention: str = "",
    station_name: str = "",
) -> bool:
    """
    Send an embed message to a Discord webhook.
    Returns True on success.
    """
    if not webhook_url or webhook_url == "YOUR_DISCORD_WEBHOOK_URL_HERE":
        print(f"[Discord] Webhook not configured — would have sent:\n{title}\n{description}")
        return False

    color_map = {"weather": 0x3498DB, "traffic": 0xE74C3C, "default": 0x2ECC71}
    embed_color = color or color_map.get("default")

    embed = {
        "title": title,
        "description": description,
        "color": embed_color,
        "footer": {
            "text": f"Source: {station_name} • {datetime.now().strftime('%b %d %Y %H:%M')}"
        },
    }

    content = mention if mention else ""
    payload = {"content": content, "embeds": [embed]}

    try:
        resp = httpx.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[Discord] Failed to send webhook: {e}")
        return False


def weather_embed(summary: str, station_name: str, webhook_url: str, mention: str = "") -> bool:
    return send_discord(
        webhook_url=webhook_url,
        title="🌤️ Weather Update",
        description=summary,
        color=0x3498DB,
        mention=mention,
        station_name=station_name,
    )


def traffic_embed(summary: str, station_name: str, webhook_url: str, mention: str = "") -> bool:
    return send_discord(
        webhook_url=webhook_url,
        title="🚗 Traffic Alert",
        description=summary,
        color=0xE74C3C,
        mention=mention,
        station_name=station_name,
    )
