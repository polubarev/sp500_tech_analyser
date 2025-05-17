import os
import re
import json
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request as flask_request
from google.cloud import storage

# Name of your GCS bucket (set this in your env)

def parse_block(container, term_name):
    """
    Given the top-level BeautifulSoup container and one of
    "Short term", "Medium term", "Long term", extract:
      - analysis (text outside comments/headings)
      - any special comment block (RSI or SR)
      - conclusion
      - recommendation + score
    """
    # find the specific div by matching its <h2>
    for div in container.find_all("div", class_="cr_oneColWith20pctMargins"):
        h2 = div.find("h2")
        if h2 and h2.get_text(strip=True) == term_name:
            # free-form analysis (everything except h2/h3 and comments)
            parts = []
            for elem in div.children:
                if getattr(elem, "name", None) in ("h2", "h3"):
                    continue
                text = elem.get_text(strip=True) if hasattr(elem, "get_text") else ""
                if text:
                    parts.append(text)
            analysis = " ".join(parts)

            html = str(div)
            # pick up either RSI or SR comment
            special = None
            if term_name.lower().startswith("short"):
                m = re.search(r"<!--RSI start-->(.*?)<!--RSI end-->", html, re.S)
                special = m.group(1).strip() if m else None
            elif term_name.lower().startswith("medium"):
                m = re.search(r"<!--SR start-->(.*?)<!--SR end-->", html, re.S)
                special = m.group(1).strip() if m else None

            # technical conclusion
            cm = re.search(r"<!--techConclusionStart-->(.*?)<!--techConclusionEnd-->", html, re.S)
            conclusion = cm.group(1).strip() if cm else None

            # recommendation label + score
            h3 = div.find("h3")
            span = h3.find("span", id=re.compile(r".*CommentaryEvaluation")) if h3 else None
            recommendation = span.get_text(strip=True) if span else None
            sm = re.search(r"Score:\s*([-+]?\d+)", h3.get_text() if h3 else "")
            score = int(sm.group(1)) if sm else None

            return {
                "analysis": analysis,
                "special": special,
                "conclusion": conclusion,
                "recommendation": recommendation,
                "score": score
            }

    return None


def parse_and_store(request):
    """
    Cloud Function entry point: fetches the page, parses
    three term-blocks, adds a timestamp, writes JSON to GCS.
    """
    URL = "https://www.investtech.com/main/market.php?CompanyID=10400521&product=241"
    resp = requests.get(URL)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # grab the overall container once
    top = soup

    data = {
        "datetime": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "short_term": parse_block(top, "Short term"),
        "medium_term": parse_block(top, "Medium term"),
        "long_term": parse_block(top, "Long term"),
    }

    # serialize
    payload = json.dumps(data, ensure_ascii=False, indent=2)

    # upload to GCS
    client = storage.Client(project="tg-bot-sso")
    bucket = client.bucket("sp-500-tech-analysis")
    # you can choose the object name format; here's a timestamped filename
    blob_name = f"investtech_{data['datetime'].replace(':','').replace('-','')}.json"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(payload, content_type="application/json")

    # return the JSON so you can inspect it in your tests
    return jsonify(data)


# --- LOCAL RUN SUPPORT ---
if __name__ == "__main__":
    app = Flask(__name__)
    # wrap the GCF handler so Flask will pass in the global `flask_request`
    def local_handler():
        return parse_and_store(flask_request)
    app.add_url_rule("/", "local_handler", local_handler, methods=["GET", "POST"])
    app.run(host="127.0.0.1", port=8080, debug=True)
