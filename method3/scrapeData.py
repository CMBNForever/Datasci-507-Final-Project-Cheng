#!/usr/bin/env python3
"""
Download the plaintext from the URLs in the retrieval corpus

Input:
data/knowledge/urls_who_samhsa.json

Outputs:
data/knowledge/scraped/text/{id}.txt
data/knowledge/scraped/manifest.json

Run using the command:
python method3/scrapeData.py
"""

import json
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
TIMEOUT = 45
DEFAULT_DELAY = 1.0
ROOT = Path(__file__).resolve().parent.parent
URLS_PATH = ROOT / "data/knowledge/urls_who_samhsa.json"
OUT_DIR = ROOT / "data/knowledge/scraped"


def html_to_text(html: bytes | str) -> str:
    """
    Transform page to text
    """
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    lines = []
    for ln in soup.get_text(separator="\n").splitlines():
        if ln.strip():
            lines.append(ln.strip())
    return "\n".join(lines)


def pdf_to_text(data: bytes) -> str:
    """
    Transform pdf to text
    """
    reader = PdfReader(BytesIO(data))
    parts: list[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text()
        except Exception:
            t = ""
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def is_pdf(url: str, content_type: str | None) -> bool:
    """
    Check if scraped is pdf
    """
    if content_type and "pdf" in content_type.lower():
        return True
    return urlparse(url).path.lower().endswith(".pdf")


def fetch(session: requests.Session, url: str) -> tuple[bytes, str | None]:
    r = session.get(url, timeout=TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    ct = r.headers.get("Content-Type", "").split(";")[0].strip() or None
    return r.content, ct


def main() -> int:
    if not URLS_PATH.is_file():
        print(f"Missing URL list: {URLS_PATH}", file=sys.stderr)
        return 1

    entries = json.loads(URLS_PATH.read_text(encoding="utf-8"))
    text_dir = OUT_DIR / "text"
    text_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    manifest: list[dict] = []
    for entry in entries:
        uid = entry["id"]
        url = entry["url"]
        rec: dict = {
            "id": uid,
            "url": url,
            "source": entry.get("source", ""),
            "ok": False,
            "error": None,
            "text_path": None,
            "text_chars": 0,
        }
        time.sleep(DEFAULT_DELAY)
        try:
            body, ctype = fetch(session, url)
            if is_pdf(url, ctype):
                text = pdf_to_text(body)
            else:
                text = html_to_text(body)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if len(text) < 80:
                rec["error"] = "text_too_short_or_empty"
                manifest.append(rec)
                print(f"[{uid}]'s text is too short")
                continue
            out = text_dir / f"{uid}.txt"
            out.write_text(text, encoding="utf-8")
            rec["ok"] = True
            rec["text_path"] = str(out.relative_to(Path(__file__).resolve().parent.parent))
            rec["text_chars"] = len(text)
            print(f"[{uid}] processed for ({len(text)} chars)")
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[{uid}] fail: {rec['error']}")
        manifest.append(rec)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ok_n = sum(1 for m in manifest if m["ok"])
    print(f"\nProcessed {ok_n}/{len(manifest)} saved under {OUT_DIR}/")
    if ok_n:
        return 0
    else:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
