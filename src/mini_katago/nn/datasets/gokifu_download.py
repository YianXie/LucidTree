from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse, unquote

import requests  # type: ignore
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter  # type:ignore
from urllib3.util.retry import Retry


BASE_URL = "http://gokifu.com/index.php"
DEFAULT_UA = (
    "Mozilla/5.0 (compatible; SGFDownloader/1.0; +https://example.com/bot-info)"
)

SGF_EXT_RE = re.compile(r"\.sgf(?:\?.*)?$", re.IGNORECASE)


@dataclass
class Config:
    out_dir: Path
    start_page: int
    end_page: int
    delay_s: float
    timeout_s: float
    max_per_page: Optional[int]


def make_session(user_agent: str = DEFAULT_UA) -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": user_agent})

    # Retries for transient errors
    retry = Retry(
        total=5,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess


def safe_filename_from_url(url: str) -> str:
    """
    Derive a reasonable filename from the URL path (keeps Unicode),
    and fall back if missing.
    """
    parsed = urlparse(url)
    name = Path(unquote(parsed.path)).name  # decode %xx, keep unicode chars
    if not name:
        name = f"download_{int(time.time() * 1000)}.sgf"
    if not name.lower().endswith(".sgf"):
        name += ".sgf"
    # Avoid path separators etc.
    name = name.replace("/", "_").replace("\\", "_")
    return name


def iter_download_links_from_page(html: str, page_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")

    links: list[str] = []
    # Find each game block
    for block in soup.select("div.player_block.cblock_3"):
        game_type = block.select_one("div.game_type")
        if not game_type:
            continue
        a_tags = game_type.find_all("a")
        if len(a_tags) < 2:
            continue

        href = a_tags[1].get("href")
        if not href:
            continue

        abs_url = urljoin(page_url, href)  # type: ignore

        # Keep only .sgf links (best-effort)
        if SGF_EXT_RE.search(abs_url):  # type: ignore
            links.append(abs_url)  # type: ignore

    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for u in links:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def fetch_text(session: requests.Session, url: str, timeout_s: float) -> str:
    r = session.get(url, timeout=timeout_s)
    r.raise_for_status()
    # requests will guess encoding; BeautifulSoup can usually handle it anyway
    return r.text  # type: ignore


def download_file(
    session: requests.Session,
    url: str,
    dest: Path,
    timeout_s: float,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Stream download to avoid holding whole file in RAM
    with session.get(url, timeout=timeout_s, stream=True) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, dest)


def page_url_for(i: int) -> str:
    """
    gokifu uses index.php with query params in some sections,
    but you can adjust this once you confirm the actual pagination pattern.

    If the site uses something like:
        index.php?page=2
    then modify this function accordingly.
    """
    # Common patterns you can try; uncomment the correct one:
    # return f"{BASE_URL}?page={i}"
    # return f"{BASE_URL}?p={i}"
    # return f"{BASE_URL}?paged={i}"

    # Fallback: page 1 is the base URL; others try ?page=
    return BASE_URL if i == 1 else f"{BASE_URL}?p={i}"


def run(cfg: Config) -> None:
    session = make_session()

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving SGFs to: {cfg.out_dir.resolve()}")

    total_found = 0
    total_downloaded = 0

    for page_num in range(cfg.start_page, cfg.end_page + 1):
        url = page_url_for(page_num)
        print(f"\n[Page {page_num}] Fetching: {url}")

        try:
            html = fetch_text(session, url, cfg.timeout_s)
        except Exception as e:
            print(f"  ! Failed to fetch page {page_num}: {e}")
            continue

        links = iter_download_links_from_page(html, url)
        if cfg.max_per_page is not None:
            links = links[: cfg.max_per_page]

        print(f"  Found {len(links)} SGF link(s)")
        total_found += len(links)

        for k, link in enumerate(links, start=1):
            filename = safe_filename_from_url(link)
            dest = cfg.out_dir / filename

            if dest.exists():
                print(f"   - [{k}/{len(links)}] Skipping (exists): {filename}")
                continue

            try:
                print(f"   + [{k}/{len(links)}] Downloading: {filename}")
                download_file(session, link, dest, cfg.timeout_s)
                total_downloaded += 1
            except Exception as e:
                print(f"     ! Download failed: {e}")
            finally:
                time.sleep(cfg.delay_s)

        # Small pause between pages too
        time.sleep(cfg.delay_s)

    print("\nDone.")
    print(f"Total links found: {total_found}")
    print(f"Total files downloaded: {total_downloaded}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Discover and download SGF files from gokifu.com list pages."
    )
    ap.add_argument(
        "--out", type=Path, default=Path("sgf_downloads"), help="Output folder"
    )
    ap.add_argument("--start", type=int, default=1, help="Start page number")
    ap.add_argument("--end", type=int, default=10, help="End page number (inclusive)")
    ap.add_argument(
        "--delay", type=float, default=1.0, help="Delay between requests (seconds)"
    )
    ap.add_argument(
        "--timeout", type=float, default=20.0, help="HTTP timeout (seconds)"
    )
    ap.add_argument(
        "--max-per-page", type=int, default=None, help="Limit downloads per page"
    )
    args = ap.parse_args()

    cfg = Config(
        out_dir=args.out,
        start_page=args.start,
        end_page=args.end,
        delay_s=args.delay,
        timeout_s=args.timeout,
        max_per_page=args.max_per_page,
    )
    run(cfg)


if __name__ == "__main__":
    main()
