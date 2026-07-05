from __future__ import annotations

import argparse
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = ROOT / "data" / "processed" / "msci_strategy" / "standard_in_out_events_all_countries.csv"
OUT_DIR = ROOT / "data" / "intermediate" / "yahoo_multicountry"
TICKER_CACHE_DIR = OUT_DIR / "ticker_cache"
MAPPING_OUT = OUT_DIR / "yahoo_ticker_mapping.csv"
FETCH_STATUS_OUT = OUT_DIR / "yahoo_price_fetch_status.csv"


MARKET_BENCHMARKS = {
    "USA": "SPY",
    "India": "^NSEI",
    "Indonesia": "^JKSE",
    "Korea": "^KS11",
}

COUNTRY_SUFFIXES = {
    "India": [".NS", ".BO"],
    "Indonesia": [".JK"],
    "Korea": [".KS", ".KQ"],
    "USA": [""],
}

COUNTRY_EXCHANGES = {
    "India": {"NSI", "BSE"},
    "Indonesia": {"JKT"},
    "Korea": {"KSC", "KOE"},
    "USA": {"NMS", "NYQ", "ASE", "PCX", "PNK"},
}

MANUAL_ALIASES = {
    ("India", "GRASIM INDUSTRIES"): "GRASIM.NS",
    ("India", "DIVI'S LABORATORIES"): "DIVISLAB.NS",
    ("India", "INDIAN OIL CORP"): "IOC.NS",
    ("India", "RURAL ELECTRIFICATION CO"): "RECLTD.NS",
    ("India", "APOLLO HOSPITALS"): "APOLLOHOSP.NS",
    ("India", "BRITANNIA INDUSTRIES"): "BRITANNIA.NS",
    ("India", "PETRONET LNG"): "PETRONET.NS",
    ("India", "VAKRANGEE"): "VAKRANGEE.NS",
    ("India", "ACC"): "ACC.NS",
    ("India", "POWER FINANCE CORP"): "PFC.NS",
    ("India", "TATA MOTORS A"): "TATAMTRDVR.NS",
    ("India", "AVENUE SUPERMARTS"): "DMART.NS",
    ("India", "INTERGLOBE AVIATION"): "INDIGO.NS",
    ("India", "PIDILITE INDUSTRIES"): "PIDILITIND.NS",
    ("India", "POWER GRID CORP OF INDIA"): "POWERGRID.NS",
    ("Indonesia", "LIPPO KARAWACI TBK"): "LPKR.JK",
    ("Indonesia", "MEDIA NUSANTARA CITRA"): "MNCN.JK",
    ("Indonesia", "SUMMARECON AGUNG"): "SMRA.JK",
    ("Indonesia", "BANK TABUNGAN NEGARA"): "BBTN.JK",
    ("Indonesia", "XL AXIATA TBK"): "EXCL.JK",
    ("Indonesia", "INDAH KIAT PULP & PAPER"): "INKP.JK",
    ("Indonesia", "AKR CORPORINDO"): "AKRA.JK",
    ("Indonesia", "MATAHARI DEPARTMENT"): "LPPF.JK",
    ("Indonesia", "WASKITA KARYA PERSERO"): "WSKT.JK",
    ("Indonesia", "BUKIT ASAM"): "PTBA.JK",
    ("Indonesia", "PABRIK K TJIWI KIMIA"): "TKIM.JK",
    ("Indonesia", "TOWER BERSAMA INFRA"): "TBIG.JK",
    ("Indonesia", "BARITO PACIFIC"): "BRPT.JK",
    ("Indonesia", "SURYA CITRA MEDIA"): "SCMA.JK",
    ("Indonesia", "ACE HARDWARE INDONESIA"): "ACES.JK",
    ("Korea", "DOOSAN BOBCAT"): "241560.KS",
    ("Korea", "MEDY-TOX"): "086900.KQ",
    ("Korea", "MEDYTOX"): "086900.KQ",
    ("Korea", "PANOCEAN"): "028670.KS",
    ("Korea", "PAN OCEAN"): "028670.KS",
    ("Korea", "LG UPLUS"): "032640.KS",
    ("Korea", "CELLTRION HEALTHCARE"): "091990.KQ",
    ("Korea", "ING LIFE INSURANCE KOREA"): "079440.KS",
    ("Korea", "SILLAJEN"): "215600.KQ",
}


def normalize_name(value: object) -> str:
    s = "" if value is None else str(value).upper()
    s = s.replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def compact_name(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", normalize_name(value))


def query_variants(security_name: str) -> list[str]:
    base = normalize_name(security_name)
    no_noise = re.sub(
        r"\b(TBK|PT|PERSERO|LTD|LIMITED|CORP|CORPORATION|COMPANY|CO|HOLDINGS|HLDGS|INC|INDIA|INDONESIA|KOREA)\b",
        " ",
        base,
    )
    no_noise = re.sub(r"\s+", " ", no_noise).strip()
    words = [w for w in base.split() if len(w) > 1]
    variants = [
        security_name,
        base,
        no_noise,
        " ".join(words[:3]),
        " ".join(words[:2]),
        words[0] if words else base,
    ]
    cleaned: list[str] = []
    for item in variants:
        item = re.sub(r"\s+", " ", str(item)).strip()
        if item and item not in cleaned:
            cleaned.append(item)
    return cleaned


def suffix_ok(country: str, symbol: str, exchange: str | None) -> bool:
    symbol = symbol.upper()
    exchange = (exchange or "").upper()
    suffixes = COUNTRY_SUFFIXES.get(country, [])
    exchanges = COUNTRY_EXCHANGES.get(country, set())
    return any(symbol.endswith(sfx) for sfx in suffixes if sfx) or exchange in exchanges


def candidate_score(country: str, security_name: str, quote: dict) -> float:
    symbol = str(quote.get("symbol", "")).upper()
    name = quote.get("shortname") or quote.get("longname") or ""
    exchange = quote.get("exchange")
    if not symbol or not suffix_ok(country, symbol, exchange):
        return -1.0
    target = compact_name(security_name)
    cand = compact_name(name)
    similarity = SequenceMatcher(None, target, cand).ratio() if target and cand else 0.0
    prefix_bonus = 0.15 if target[:5] and cand.startswith(target[:5]) else 0.0
    suffix_bonus = 0.10 if any(symbol.endswith(sfx) for sfx in COUNTRY_SUFFIXES.get(country, [])[:1] if sfx) else 0.0
    return similarity + prefix_bonus + suffix_bonus


def search_yahoo(country: str, security_name: str, sleep_sec: float = 0.15) -> dict:
    manual = MANUAL_ALIASES.get((country, security_name))
    if manual:
        return {
            "country": country,
            "security_name": security_name,
            "ticker": manual,
            "status": "mapped_manual_alias",
            "score": 1.0,
            "query_used": "manual_alias",
            "candidate_name": "",
            "exchange": "",
            "reason": "",
        }

    best: dict | None = None
    best_score = -1.0
    best_query = ""
    saw_quotes = False

    for query in query_variants(security_name):
        try:
            result = yf.Search(query, max_results=12)
            quotes = result.quotes or []
        except Exception as exc:
            return {
                "country": country,
                "security_name": security_name,
                "ticker": None,
                "status": "search_error",
                "score": None,
                "query_used": query,
                "candidate_name": "",
                "exchange": "",
                "reason": f"{type(exc).__name__}: {exc}",
            }

        if quotes:
            saw_quotes = True
        for quote in quotes:
            score = candidate_score(country, security_name, quote)
            if score > best_score:
                best = quote
                best_score = score
                best_query = query
        if best is not None and best_score >= 0.72:
            break
        time.sleep(sleep_sec)

    if best is None or best_score < 0.45:
        return {
            "country": country,
            "security_name": security_name,
            "ticker": None,
            "status": "no_country_candidate" if saw_quotes else "no_search_result",
            "score": best_score if best_score >= 0 else None,
            "query_used": best_query,
            "candidate_name": "",
            "exchange": "",
            "reason": "No Yahoo quote matched the expected country suffix/exchange with sufficient name similarity.",
        }

    return {
        "country": country,
        "security_name": security_name,
        "ticker": str(best.get("symbol", "")).upper(),
        "status": "mapped_yahoo_search",
        "score": round(float(best_score), 4),
        "query_used": best_query,
        "candidate_name": best.get("shortname") or best.get("longname") or "",
        "exchange": best.get("exchange") or "",
        "reason": "",
    }


def safe_cache_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_").replace("^", "INDEX_")
    return TICKER_CACHE_DIR / f"{safe}.csv"


def download_ticker(ticker: str, events: pd.DataFrame) -> dict:
    start = pd.to_datetime(events["announce_date"], errors="coerce").min() - pd.Timedelta(days=60)
    end = pd.to_datetime(events["effective_date"], errors="coerce").max() + pd.Timedelta(days=20)
    path = safe_cache_path(ticker)

    if path.exists():
        cached = pd.read_csv(path)
        if not cached.empty:
            return {"ticker": ticker, "status": "cached", "rows": len(cached), "path": str(path), "reason": ""}

    try:
        px = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=False,
            actions=False,
            progress=False,
            group_by="column",
            threads=False,
        )
    except Exception as exc:
        return {"ticker": ticker, "status": "download_error", "rows": 0, "path": str(path), "reason": f"{type(exc).__name__}: {exc}"}

    if px is None or px.empty:
        return {"ticker": ticker, "status": "no_price_data", "rows": 0, "path": str(path), "reason": "Yahoo returned no historical rows."}

    if isinstance(px.columns, pd.MultiIndex):
        px.columns = [c[0] for c in px.columns]
    px = px.reset_index()
    if "Date" in px.columns:
        px = px.rename(columns={"Date": "date"})
    elif "Datetime" in px.columns:
        px = px.rename(columns={"Datetime": "date"})
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px["ticker"] = ticker
    keep = [c for c in ["date", "ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in px.columns]
    px = px[keep].dropna(subset=["date"]).sort_values("date")
    path.parent.mkdir(parents=True, exist_ok=True)
    px.to_csv(path, index=False)
    return {"ticker": ticker, "status": "downloaded", "rows": len(px), "path": str(path), "reason": ""}


def save_fetch_status(fetch_rows: list[dict]) -> pd.DataFrame:
    new = pd.DataFrame(fetch_rows)
    if new.empty:
        return new
    if FETCH_STATUS_OUT.exists():
        existing = pd.read_csv(FETCH_STATUS_OUT)
        if "ticker" in existing.columns:
            existing = existing[~existing["ticker"].isin(new["ticker"])]
        fetch = pd.concat([existing, new], ignore_index=True, sort=False)
    else:
        fetch = new
    fetch.to_csv(FETCH_STATUS_OUT, index=False)
    return fetch


def download_benchmarks(events: pd.DataFrame, countries: list[str], sleep_sec: float) -> list[dict]:
    fetch_rows: list[dict] = []
    for country in countries:
        benchmark = MARKET_BENCHMARKS.get(country)
        if not benchmark:
            continue
        country_events = events[events["country"].eq(country)].copy()
        if country_events.empty:
            continue
        status = download_ticker(benchmark, country_events)
        status.update({"country": country, "asset_type": "benchmark"})
        fetch_rows.append(status)
        print(f"{country} benchmark {benchmark}: {status['status']} ({status['rows']} rows)")
        time.sleep(sleep_sec)
    return fetch_rows


def build_mapping(events: pd.DataFrame, sleep_sec: float) -> pd.DataFrame:
    if MAPPING_OUT.exists():
        existing = pd.read_csv(MAPPING_OUT)
    else:
        existing = pd.DataFrame()

    done = set()
    rows: list[dict] = []
    if not existing.empty:
        for _, row in existing.iterrows():
            done.add((row["country"], row["security_name"]))
        rows.extend(existing.to_dict("records"))

    universe = (
        events[["country", "security_name", "ticker"]]
        .drop_duplicates(["country", "security_name"])
        .sort_values(["country", "security_name"])
    )

    for _, row in universe.iterrows():
        country = row["country"]
        security_name = row["security_name"]
        if (country, security_name) in done:
            continue
        if pd.notna(row.get("ticker")) and str(row["ticker"]).strip():
            rows.append(
                {
                    "country": country,
                    "security_name": security_name,
                    "ticker": str(row["ticker"]).strip(),
                    "status": "existing_repo_mapping",
                    "score": 1.0,
                    "query_used": "existing_repo_mapping",
                    "candidate_name": "",
                    "exchange": "",
                    "reason": "",
                }
            )
            continue
        if country not in COUNTRY_SUFFIXES or country == "USA":
            rows.append(
                {
                    "country": country,
                    "security_name": security_name,
                    "ticker": None,
                    "status": "unsupported_country",
                    "score": None,
                    "query_used": "",
                    "candidate_name": "",
                    "exchange": "",
                    "reason": "No country suffix rules configured.",
                }
            )
            continue
        mapped = search_yahoo(country, security_name, sleep_sec=sleep_sec)
        rows.append(mapped)
        if len(rows) % 25 == 0:
            pd.DataFrame(rows).to_csv(MAPPING_OUT, index=False)
            print(f"Mapped {len(rows)} unique securities...")

    out = pd.DataFrame(rows)
    out.to_csv(MAPPING_OUT, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Map MSCI event names to Yahoo tickers and cache Yahoo price/volume data.")
    parser.add_argument("--sleep-sec", type=float, default=0.15)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--benchmarks-only", action="store_true", help="Only cache local market benchmark/index series.")
    parser.add_argument(
        "--countries",
        nargs="*",
        default=["India", "Indonesia", "Korea"],
        help="Countries to download after mapping. Defaults to non-U.S. markets.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TICKER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(EVENTS_PATH)
    if args.benchmarks_only:
        fetch = save_fetch_status(download_benchmarks(events, args.countries, args.sleep_sec))
        print("Fetch status:")
        print(fetch["status"].value_counts(dropna=False).to_string() if not fetch.empty else "No benchmark rows fetched.")
        print(f"Saved fetch status -> {FETCH_STATUS_OUT}")
        return

    mapping = build_mapping(events, sleep_sec=args.sleep_sec)
    print("Mapping status:")
    print(mapping["status"].value_counts(dropna=False).to_string())

    if args.skip_download:
        return

    mapped = mapping[mapping["ticker"].notna() & mapping["country"].isin(args.countries)].copy()
    fetch_rows = []
    for ticker, group in mapped.groupby("ticker"):
        event_rows = events.merge(group[["country", "security_name", "ticker"]], on=["country", "security_name"], how="inner")
        status = download_ticker(str(ticker), event_rows)
        status.update(
            {
                "country": ";".join(sorted(group["country"].dropna().astype(str).unique())),
                "asset_type": "equity",
            }
        )
        fetch_rows.append(status)
        print(f"{ticker}: {status['status']} ({status['rows']} rows)")
        time.sleep(args.sleep_sec)

    fetch_rows.extend(download_benchmarks(events, args.countries, args.sleep_sec))
    fetch = save_fetch_status(fetch_rows)
    print("Fetch status:")
    print(fetch["status"].value_counts(dropna=False).to_string())
    print(f"Saved mapping -> {MAPPING_OUT}")
    print(f"Saved fetch status -> {FETCH_STATUS_OUT}")


if __name__ == "__main__":
    main()
