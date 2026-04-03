"""Seed caption style presets to match OpusClip's 10+ templates."""
import sqlite3
import uuid
from datetime import datetime

DB_PATH = "storage/video_extract_pro.db"

STYLES = [
    {
        "name": "Classic White Karaoke",
        "style_type": "KARAOKE",
        "font_family": "Arial",
        "font_size": 56,
        "font_weight": "bold",
        "text_color": "#FFFFFF",
        "highlight_color": "#FFFF00",
        "background_color": None,
        "stroke_color": "#000000",
        "stroke_width": 3,
        "position": "BOTTOM",
        "margin_bottom": 50,
        "animation_duration": 0.1,
        "words_per_line": 6,
    },
    {
        "name": "Neon Green Pop",
        "style_type": "BOUNCE",
        "font_family": "Impact",
        "font_size": 64,
        "font_weight": "bold",
        "text_color": "#FFFFFF",
        "highlight_color": "#00FF41",
        "background_color": None,
        "stroke_color": "#000000",
        "stroke_width": 4,
        "position": "CENTER",
        "margin_bottom": 80,
        "animation_duration": 0.15,
        "words_per_line": 4,
    },
    {
        "name": "Sunset Glow",
        "style_type": "FADE",
        "font_family": "Georgia",
        "font_size": 52,
        "font_weight": "bold",
        "text_color": "#FF6B35",
        "highlight_color": "#FFD700",
        "background_color": None,
        "stroke_color": "#1A1A2E",
        "stroke_width": 3,
        "position": "BOTTOM",
        "margin_bottom": 60,
        "animation_duration": 0.25,
        "words_per_line": 5,
    },
    {
        "name": "Bold Typewriter",
        "style_type": "TYPEWRITER",
        "font_family": "Courier New",
        "font_size": 48,
        "font_weight": "bold",
        "text_color": "#00FF00",
        "highlight_color": "#00FF00",
        "background_color": "#000000",
        "stroke_color": "#003300",
        "stroke_width": 1,
        "position": "BOTTOM",
        "margin_bottom": 40,
        "animation_duration": 0.04,
        "words_per_line": 8,
    },
    {
        "name": "Minimal Clean",
        "style_type": "STATIC",
        "font_family": "Helvetica",
        "font_size": 44,
        "font_weight": "normal",
        "text_color": "#FFFFFF",
        "highlight_color": "#FFFFFF",
        "background_color": None,
        "stroke_color": "#333333",
        "stroke_width": 2,
        "position": "BOTTOM",
        "margin_bottom": 50,
        "animation_duration": 0.1,
        "words_per_line": 7,
    },
    {
        "name": "TikTok Viral",
        "style_type": "KARAOKE",
        "font_family": "Arial Black",
        "font_size": 68,
        "font_weight": "bold",
        "text_color": "#FFFFFF",
        "highlight_color": "#FF0050",
        "background_color": None,
        "stroke_color": "#000000",
        "stroke_width": 4,
        "position": "CENTER",
        "margin_bottom": 100,
        "animation_duration": 0.1,
        "words_per_line": 3,
    },
    {
        "name": "YouTube Professional",
        "style_type": "KARAOKE",
        "font_family": "Roboto",
        "font_size": 50,
        "font_weight": "bold",
        "text_color": "#FFFFFF",
        "highlight_color": "#1DA1F2",
        "background_color": None,
        "stroke_color": "#0D1117",
        "stroke_width": 3,
        "position": "BOTTOM",
        "margin_bottom": 55,
        "animation_duration": 0.1,
        "words_per_line": 6,
    },
    {
        "name": "Instagram Aesthetic",
        "style_type": "FADE",
        "font_family": "Playfair Display",
        "font_size": 54,
        "font_weight": "bold",
        "text_color": "#FAFAFA",
        "highlight_color": "#E1306C",
        "background_color": None,
        "stroke_color": "#2D2D2D",
        "stroke_width": 2,
        "position": "CENTER",
        "margin_bottom": 90,
        "animation_duration": 0.3,
        "words_per_line": 4,
    },
    {
        "name": "Podcast Highlight",
        "style_type": "KARAOKE",
        "font_family": "Trebuchet MS",
        "font_size": 58,
        "font_weight": "bold",
        "text_color": "#FFFFFF",
        "highlight_color": "#9B59B6",
        "background_color": None,
        "stroke_color": "#2C3E50",
        "stroke_width": 3,
        "position": "BOTTOM",
        "margin_bottom": 45,
        "animation_duration": 0.1,
        "words_per_line": 5,
    },
    {
        "name": "Gaming Neon",
        "style_type": "BOUNCE",
        "font_family": "Impact",
        "font_size": 72,
        "font_weight": "bold",
        "text_color": "#00FFFF",
        "highlight_color": "#FF00FF",
        "background_color": None,
        "stroke_color": "#000000",
        "stroke_width": 5,
        "position": "CENTER",
        "margin_bottom": 70,
        "animation_duration": 0.12,
        "words_per_line": 3,
    },
    {
        "name": "News Ticker",
        "style_type": "TYPEWRITER",
        "font_family": "Arial",
        "font_size": 46,
        "font_weight": "bold",
        "text_color": "#FFFFFF",
        "highlight_color": "#FF4444",
        "background_color": "#CC0000",
        "stroke_color": "#000000",
        "stroke_width": 1,
        "position": "BOTTOM",
        "margin_bottom": 30,
        "animation_duration": 0.03,
        "words_per_line": 10,
    },
    {
        "name": "Cinematic Gold",
        "style_type": "FADE",
        "font_family": "Garamond",
        "font_size": 60,
        "font_weight": "bold",
        "text_color": "#FFD700",
        "highlight_color": "#FFA500",
        "background_color": None,
        "stroke_color": "#1C1C1C",
        "stroke_width": 3,
        "position": "BOTTOM",
        "margin_bottom": 65,
        "animation_duration": 0.35,
        "words_per_line": 5,
    },
]


def main():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT name FROM caption_styles")
    existing = {r[0] for r in c.fetchall()}
    print(f"Existing styles: {existing}")

    now = datetime.utcnow().isoformat()
    inserted = 0

    for s in STYLES:
        if s["name"] in existing:
            print(f"  SKIP {s['name']} (exists)")
            continue

        sid = str(uuid.uuid4())
        c.execute(
            """INSERT INTO caption_styles
            (id, created_at, updated_at, name, style_type, font_family, font_size, font_weight,
             text_color, highlight_color, background_color, stroke_color, stroke_width,
             position, margin_bottom, animation_duration, words_per_line, is_default)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
            (
                sid, now, now, s["name"], s["style_type"], s["font_family"],
                s["font_size"], s["font_weight"], s["text_color"], s["highlight_color"],
                s["background_color"], s["stroke_color"], s["stroke_width"],
                s["position"], s["margin_bottom"], s["animation_duration"],
                s["words_per_line"],
            ),
        )
        inserted += 1
        print(f"  + {s['name']}")

    conn.commit()
    print(f"\nInserted {inserted} new caption styles")

    c.execute("SELECT count(*) FROM caption_styles")
    print(f"Total styles: {c.fetchone()[0]}")
    conn.close()


if __name__ == "__main__":
    main()
