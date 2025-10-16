from pathlib import Path
import logging, sys, traceback
import requests
from requests.auth import HTTPBasicAuth

OUT = Path("/workspace/data/external")
LOG_DIR = Path("/workspace/.logs")
OUT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_DIR / "update_data.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def read_secret(name: str) -> str:
    p = Path("/run/secrets") / name
    if not p.exists():
        raise SystemExit(f"Missing secret file: /run/secrets/{name}")
    return p.read_text().strip()

BASE = (Path("/run/secrets/ASHILD_BASE").read_text().strip()
        if (Path("/run/secrets/ASHILD_BASE").exists())
        else "https://www.ashild.se/Api/Export")
USER = read_secret("ASHILD_USER")
PASS = read_secret("ASHILD_PASS")

ENDPOINTS = {
    "customers":    f"{BASE}/Customers",
    "transactions": f"{BASE}/Transactions",
    "products":     f"{BASE}/Articles",
}

def dl_csv(name: str, url: str) -> None:
    out_path = OUT / f"{name}.csv"
    logging.info(f"Downloading {name} from {url} -> {out_path.name}")
    with requests.get(url, auth=HTTPBasicAuth(USER, PASS), timeout=120, stream=True) as r:
        if r.status_code == 401:
            raise RuntimeError(f"401 Unauthorized for {url}. Check secrets.")
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(1 << 20):  # 1 MiB chunks
                if chunk:
                    f.write(chunk)
    logging.info(f"Saved {out_path.name}")

def main() -> int:
    try:
        for name, url in ENDPOINTS.items():
            dl_csv(name, url)
        logging.info("All endpoints fetched successfully.")
        return 0
    except Exception as e:
        logging.error("Update failed: %s", e)
        logging.error("Traceback:\n%s", traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())