from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TMPDIR", "/tmp")) / "msci_matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

try:
    import pdfplumber
except ImportError:  # pragma: no cover - handled at runtime
    pdfplumber = None


DATA_DIR = ROOT / "data"
PDF_ROOT = DATA_DIR / "msci_public_lists"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
OUT_DIR = DATA_DIR / "processed" / "msci_strategy"
CHART_DIR = OUT_DIR / "charts"

USA_OCR_EVENTS = INTERMEDIATE_DIR / "msci_usa_events_ocr.csv"
USA_STANDARD_EVENTS = INTERMEDIATE_DIR / "msci_standard_in_out_events_usa.csv"
CACHE_DIR = INTERMEDIATE_DIR / "yahoo_event_windows" / "ticker_cache"
EVENT_WINDOW_PANEL = INTERMEDIATE_DIR / "yahoo_event_windows" / "event_window_panel.csv"
MULTICOUNTRY_YAHOO_DIR = INTERMEDIATE_DIR / "yahoo_multicountry"
MULTICOUNTRY_CACHE_DIR = MULTICOUNTRY_YAHOO_DIR / "ticker_cache"
MULTICOUNTRY_MAPPING_PATH = MULTICOUNTRY_YAHOO_DIR / "yahoo_ticker_mapping.csv"

MARKET_BENCHMARKS = {
    "USA": "SPY",
    "India": "^NSEI",
    "Indonesia": "^JKSE",
    "Korea": "^KS11",
}

MARKET_DETAILS = {
    "USA": {
        "stock_market": "United States equities",
        "primary_exchanges": "NYSE/Nasdaq/NYSE American",
        "yahoo_suffixes": "none",
        "benchmark_ticker": "SPY",
    },
    "India": {
        "stock_market": "India equities",
        "primary_exchanges": "NSE/BSE",
        "yahoo_suffixes": ".NS/.BO",
        "benchmark_ticker": "^NSEI",
    },
    "Indonesia": {
        "stock_market": "Indonesia equities",
        "primary_exchanges": "Indonesia Stock Exchange",
        "yahoo_suffixes": ".JK",
        "benchmark_ticker": "^JKSE",
    },
    "Korea": {
        "stock_market": "Korea equities",
        "primary_exchanges": "KOSPI/KOSDAQ",
        "yahoo_suffixes": ".KS/.KQ",
        "benchmark_ticker": "^KS11",
    },
}


TARGET_COUNTRIES = {
    "INDONESIA": "Indonesia",
    "KOREA": "Korea",
    "INDIA": "India",
    "USA": "USA",
    "UNITED STATES": "USA",
}

TIER_RANK = {
    "none": 0,
    "micro_cap": 1,
    "small_cap": 2,
    "standard": 3,
}

MONTH_ABBR = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

MONTH_FULL = {
    "JANUARY": 1,
    "FEBRUARY": 2,
    "MARCH": 3,
    "APRIL": 4,
    "MAY": 5,
    "JUNE": 6,
    "JULY": 7,
    "AUGUST": 8,
    "SEPTEMBER": 9,
    "OCTOBER": 10,
    "NOVEMBER": 11,
    "DECEMBER": 12,
}

DATE_FROM_FILE_RE = re.compile(r"MSCI_([A-Za-z]{3})(\d{2})_")
HEADER_RE = re.compile(r"^\s*MSCI\s+(.+?)\s+INDEX\b", re.IGNORECASE)
ANNOUNCE_RE = re.compile(
    r"\bGENEVA\s*,?\s+("
    + "|".join(MONTH_FULL)
    + r")\s+(\d{1,2})\s*,\s*(\d{4})\b",
    re.IGNORECASE,
)
EFFECTIVE_RE = re.compile(
    r"\bCLOSE\s+OF\s+("
    + "|".join(MONTH_FULL)
    + r")\s+(\d{1,2})\s*,\s*(\d{4})\b",
    re.IGNORECASE,
)
EVENT_TICKER_RE = re.compile(r"\(([A-Z]+)\s*:\s*([^)]+)\)")


@dataclass(frozen=True)
class StrategyRule:
    anchor: str
    entry_offset: int
    exit_offset: int

    @property
    def label(self) -> str:
        return f"{self.anchor}_d{self.entry_offset:+d}_to_d{self.exit_offset:+d}"


def clean_text(value: object) -> str:
    s = "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_security_name(value: object) -> str:
    s = clean_text(value).upper()
    s = re.sub(r"^[+\-]\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" ,.;:")


def normalize_country(value: object) -> str | None:
    s = normalize_security_name(value)
    return TARGET_COUNTRIES.get(s)


def normalize_tier(value: object) -> str:
    s = clean_text(value).lower().replace("-", "_")
    if s in {"", "nan", "na", "n/a", "none"}:
        return "none"
    if s in {"global_standard", "global standard", "standard", "standards", "std", "st"}:
        return "standard"
    if s in {"small", "smallcap", "small_cap", "small cap", "sc"}:
        return "small_cap"
    if s in {"micro", "microcap", "micro_cap", "micro cap"}:
        return "micro_cap"
    return s


def date_from_filename(path: Path) -> str | None:
    match = DATE_FROM_FILE_RE.search(path.name)
    if not match:
        return None
    mon, yy = match.groups()
    month = MONTH_ABBR.get(mon.upper())
    if not month:
        return None
    return f"{2000 + int(yy):04d}-{month:02d}-01"


def bucket_from_filename(path: Path) -> str:
    name = path.name.upper()
    if "MICROPUBLICLIST" in name or "_MICRO" in name:
        return "micro_cap"
    if "SCPUBLICLIST" in name or "_SC" in name:
        return "small_cap"
    if "STPUBLICLIST" in name or "_ST" in name:
        return "standard"
    return "unknown"


def iso_from_month(month_name: str, day: str, year: str) -> str:
    return f"{int(year):04d}-{MONTH_FULL[month_name.upper()]:02d}-{int(day):02d}"


def parse_announce_effective(text: str) -> tuple[str | None, str | None]:
    compact = " ".join((text or "").split())
    announce = None
    effective = None
    match = ANNOUNCE_RE.search(compact)
    if match:
        announce = iso_from_month(match.group(1), match.group(2), match.group(3))
    match = EFFECTIVE_RE.search(compact)
    if match:
        effective = iso_from_month(match.group(1), match.group(2), match.group(3))
    return announce, effective


def is_noise_name(value: object) -> bool:
    s = normalize_security_name(value)
    if not s or len(s) < 2:
        return True
    region_labels = {
        "AMERICAS",
        "EUROPE",
        "EUROPE, MIDDLE EAST AND AFRICA",
        "MIDDLE EAST AND AFRICA",
        "ASIA",
        "ASIA PACIFIC",
        "PACIFIC",
    }
    if s in region_labels:
        return True
    if s in {"NONE", "(NONE)", "ADDITIONS", "DELETIONS"}:
        return True
    if s.startswith("PAGE ") or s.startswith("MSCI ") or "ALL RIGHTS RESERVED" in s:
        return True
    if any(term in s for term in ["WWW.MSCI.COM", "DISCLAIMER", "TRADEMARK"]):
        return True
    return False


def action_from_change(change_type: object) -> str | None:
    ct = clean_text(change_type).upper()
    if ct in {"ADD", "ADDITION", "ADDITIONS", "PROMOTION"}:
        return "ADD"
    if ct in {"DEL", "DELETE", "DELETION", "DELETIONS", "DEMOTION"}:
        return "DEL"
    return None


def classify_move(from_tier: object, to_tier: object) -> tuple[str, list[str]]:
    flags: list[str] = []
    frm = normalize_tier(from_tier)
    to = normalize_tier(to_tier)
    if frm not in TIER_RANK:
        flags.append(f"unknown_from({frm})")
    if to not in TIER_RANK:
        flags.append(f"unknown_to({to})")
    rf = TIER_RANK.get(frm, -999)
    rt = TIER_RANK.get(to, -999)
    if frm == "none" and to != "none":
        return "ADD", flags
    if frm != "none" and to == "none":
        return "DEL", flags
    if rf == -999 or rt == -999:
        return "MOVE", flags + ["unranked_tier"]
    if rt > rf:
        return "PROMOTION", flags
    if rt < rf:
        return "DEMOTION", flags
    return "MOVE", flags + ["no_tier_change"]


def from_to_for_action(action: str, bucket: str) -> tuple[str, str]:
    bucket = normalize_tier(bucket)
    if action == "ADD":
        return "none", bucket
    if action == "DEL":
        return bucket, "none"
    return "none", "none"


def group_visual_lines(words: list[dict]) -> list[list[dict]]:
    lines: dict[int, list[dict]] = {}
    for word in words:
        y = int(round(float(word["top"])))
        lines.setdefault(y, []).append(word)
    return [sorted(lines[y], key=lambda w: float(w["x0"])) for y in sorted(lines)]


def line_text(words: list[dict]) -> str:
    return clean_text(" ".join(str(w["text"]) for w in words))


def country_from_header(text: str) -> str | None:
    match = HEADER_RE.search(text)
    if not match:
        return None
    return normalize_country(match.group(1))


def parse_pdf_sections(pdf_path: Path) -> list[dict]:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is required to parse MSCI public-list PDFs.")

    event_month = date_from_filename(pdf_path)
    if event_month is None:
        return []

    bucket = bucket_from_filename(pdf_path)
    rows: list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        first_text = pdf.pages[0].extract_text() if pdf.pages else ""
        announce_date, effective_date = parse_announce_effective(first_text or "")

        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words() or []
            current_country: str | None = None
            split_x: float | None = None
            section_started = False

            for wline in group_visual_lines(words):
                text = line_text(wline)
                upper = text.upper()
                header_match = HEADER_RE.search(text)
                if header_match:
                    current_country = normalize_country(header_match.group(1))
                    split_x = None
                    section_started = False
                    continue

                if current_country is None:
                    continue
                if "ADDITIONS" in upper and "DELETIONS" in upper:
                    deletion_word = next(
                        (w for w in wline if str(w["text"]).strip().upper().startswith("DELETION")),
                        None,
                    )
                    split_x = float(deletion_word["x0"]) if deletion_word else float(page.width) / 2.0
                    section_started = True
                    continue
                if not section_started:
                    continue
                if upper.startswith("PAGE ") or "ALL RIGHTS RESERVED" in upper:
                    continue

                split = split_x if split_x is not None else float(page.width) / 2.0
                left = normalize_security_name(" ".join(str(w["text"]) for w in wline if float(w["x0"]) < split))
                right = normalize_security_name(" ".join(str(w["text"]) for w in wline if float(w["x0"]) >= split))

                for name, action in [(left, "ADD"), (right, "DEL")]:
                    if is_noise_name(name):
                        continue
                    frm, to = from_to_for_action(action, bucket)
                    rows.append(
                        {
                            "event_month": event_month,
                            "announce_date": announce_date,
                            "effective_date": effective_date,
                            "country": current_country,
                            "security_name_raw": name,
                            "security_name": normalize_security_name(name),
                            "leg_action": action,
                            "from_tier": frm,
                            "to_tier": to,
                            "bucket": bucket,
                            "source_file": pdf_path.name,
                            "source_page": page_num,
                            "source_method": "pdfplumber_words",
                            "quality_flags": "" if bucket != "unknown" else "unknown_source_bucket",
                        }
                    )

    return rows


def load_usa_ocr_legs() -> pd.DataFrame:
    if not USA_OCR_EVENTS.exists():
        return pd.DataFrame()
    df = pd.read_csv(USA_OCR_EVENTS)
    df = df[df["Country"].astype(str).str.upper().eq("USA")].copy()
    df["event_month"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["country"] = "USA"
    df["security_name_raw"] = df["Security_Name"].map(normalize_security_name)
    df["security_name"] = df["security_name_raw"]
    df["leg_action"] = df["Change_Type"].map(action_from_change)
    df["from_tier"] = df["from"].map(normalize_tier)
    df["to_tier"] = df["to"].map(normalize_tier)
    df["bucket"] = np.where(df["leg_action"].eq("ADD"), df["to_tier"], df["from_tier"])
    df["source_page"] = df.get("page", np.nan)
    df["source_method"] = "existing_usa_ocr"
    df["quality_flags"] = ""
    keep = [
        "event_month",
        "announce_date",
        "effective_date",
        "country",
        "security_name_raw",
        "security_name",
        "leg_action",
        "from_tier",
        "to_tier",
        "bucket",
        "source_file",
        "source_page",
        "source_method",
        "quality_flags",
    ]
    return df[df["leg_action"].isin(["ADD", "DEL"])][keep].copy()


def build_raw_legs(use_existing_usa: bool = True) -> pd.DataFrame:
    pdf_rows: list[dict] = []
    for pdf_path in sorted(PDF_ROOT.rglob("MSCI_*.pdf")):
        # The China A files do not contain the target country Standard/Small/Micro lists.
        if "CHINAAPUBLICLIST" in pdf_path.name.upper():
            continue
        pdf_rows.extend(parse_pdf_sections(pdf_path))

    pdf_legs = pd.DataFrame(pdf_rows)
    if use_existing_usa:
        pdf_legs = pdf_legs[pdf_legs["country"].ne("USA")].copy() if not pdf_legs.empty else pdf_legs
        usa_legs = load_usa_ocr_legs()
        legs = pd.concat([pdf_legs, usa_legs], ignore_index=True)
    else:
        legs = pdf_legs

    if legs.empty:
        return legs

    for col in ["event_month", "announce_date", "effective_date"]:
        legs[col] = pd.to_datetime(legs[col], errors="coerce").dt.strftime("%Y-%m-%d")
    legs["country"] = legs["country"].astype(str)
    legs["security_name"] = legs["security_name"].map(normalize_security_name)
    legs = legs[~legs["security_name"].map(is_noise_name)].copy()
    legs = legs.drop_duplicates(
        subset=[
            "event_month",
            "country",
            "security_name",
            "leg_action",
            "bucket",
            "source_file",
            "source_page",
        ]
    )
    return legs.sort_values(["event_month", "country", "security_name", "leg_action"]).reset_index(drop=True)


def reconcile_legs(legs: pd.DataFrame) -> pd.DataFrame:
    if legs.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    key_cols = ["event_month", "country", "security_name"]
    for (event_month, country, security_name), group in legs.groupby(key_cols, dropna=False):
        flags: list[str] = []
        adds = {normalize_tier(x) for x in group.loc[group["leg_action"].eq("ADD"), "bucket"].dropna()}
        dels = {normalize_tier(x) for x in group.loc[group["leg_action"].eq("DEL"), "bucket"].dropna()}
        bad_adds = sorted(x for x in adds if x not in TIER_RANK)
        bad_dels = sorted(x for x in dels if x not in TIER_RANK)
        if bad_adds:
            flags.append(f"unknown_add_tier({bad_adds})")
        if bad_dels:
            flags.append(f"unknown_del_tier({bad_dels})")
        adds = {x for x in adds if x in TIER_RANK and x != "none"}
        dels = {x for x in dels if x in TIER_RANK and x != "none"}
        if len(adds) > 1:
            flags.append(f"multi_add({sorted(adds)})")
        if len(dels) > 1:
            flags.append(f"multi_del({sorted(dels)})")

        announce_vals = sorted(v for v in group["announce_date"].dropna().astype(str).unique() if v != "NaT")
        effective_vals = sorted(v for v in group["effective_date"].dropna().astype(str).unique() if v != "NaT")
        announce_date = announce_vals[0] if announce_vals else None
        effective_date = effective_vals[0] if effective_vals else None
        if len(announce_vals) > 1:
            flags.append(f"inconsistent_announce_date({announce_vals})")
        if len(effective_vals) > 1:
            flags.append(f"inconsistent_effective_date({effective_vals})")

        source_file = ";".join(sorted(group["source_file"].dropna().astype(str).unique()))
        source_method = ";".join(sorted(group["source_method"].dropna().astype(str).unique()))
        quality = ";".join(sorted(x for x in group["quality_flags"].dropna().astype(str).unique() if x))

        candidates: list[tuple[str, str]]
        if len(adds) == 1 and len(dels) == 1:
            candidates = [(next(iter(dels)), next(iter(adds)))]
        elif len(adds) == 1 and len(dels) == 0:
            candidates = [("none", next(iter(adds)))]
        elif len(adds) == 0 and len(dels) == 1:
            candidates = [(next(iter(dels)), "none")]
        else:
            candidates = [("none", "none")]
            flags.append("ambiguous_legs")

        for frm, to in candidates:
            change_type, move_flags = classify_move(frm, to)
            all_flags = [x for x in flags + move_flags + ([quality] if quality else []) if x]
            rows.append(
                {
                    "event_month": event_month,
                    "announce_date": announce_date,
                    "effective_date": effective_date,
                    "country": country,
                    "security_name": security_name,
                    "change_type": change_type,
                    "from_tier": frm,
                    "to_tier": to,
                    "source_file": source_file,
                    "source_method": source_method,
                    "quality_flags": ";".join(all_flags),
                }
            )

    events = pd.DataFrame(rows)
    events = events.sort_values(["event_month", "country", "security_name"]).reset_index(drop=True)
    events.insert(0, "event_id", np.arange(1, len(events) + 1))
    return events


def parse_event_ticker(value: object) -> str | None:
    s = clean_text(value)
    match = EVENT_TICKER_RE.search(s)
    if not match:
        return None
    ticker = match.group(2).strip().replace(" ", "").rstrip("*")
    return ticker or None


def attach_existing_usa_tickers(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["security_ticker_raw"] = pd.NA
    events["ticker"] = pd.NA
    if not USA_STANDARD_EVENTS.exists():
        return events

    usa = pd.read_csv(USA_STANDARD_EVENTS)
    usa_event_month = pd.to_datetime(usa["Date"], errors="coerce", format="%m/%d/%y")
    usa["event_month"] = usa_event_month.dt.strftime("%Y-%m-%d")
    usa["country"] = "USA"
    usa["security_name"] = usa["Security_Name"].map(normalize_security_name)
    usa["security_ticker_raw"] = usa.get("Security_Ticker", pd.NA)
    usa["ticker"] = usa["security_ticker_raw"].map(parse_event_ticker)
    mapping = usa[["event_month", "country", "security_name", "security_ticker_raw", "ticker"]].drop_duplicates()
    events = events.merge(mapping, on=["event_month", "country", "security_name"], how="left", suffixes=("", "_map"))
    for col in ["security_ticker_raw", "ticker"]:
        map_col = f"{col}_map"
        if map_col in events.columns:
            events[col] = events[col].combine_first(events[map_col])
            events = events.drop(columns=[map_col])
    return events


def attach_multicountry_yahoo_tickers(events: pd.DataFrame) -> pd.DataFrame:
    if not MULTICOUNTRY_MAPPING_PATH.exists():
        return events
    mapping = pd.read_csv(MULTICOUNTRY_MAPPING_PATH)
    mapping = mapping[mapping["ticker"].notna()].copy()
    if mapping.empty:
        return events
    mapping["security_name"] = mapping["security_name"].map(normalize_security_name)
    mapping = mapping[["country", "security_name", "ticker", "status", "score", "candidate_name", "exchange"]].drop_duplicates(
        ["country", "security_name"]
    )
    mapping = mapping.rename(
        columns={
            "ticker": "yahoo_ticker",
            "status": "ticker_mapping_status",
            "score": "ticker_mapping_score",
            "candidate_name": "ticker_candidate_name",
            "exchange": "ticker_exchange",
        }
    )
    out = events.merge(mapping, on=["country", "security_name"], how="left")
    out["ticker"] = out["ticker"].combine_first(out["yahoo_ticker"])
    out["security_ticker_raw"] = out["security_ticker_raw"].combine_first(out["yahoo_ticker"])
    return out.drop(columns=["yahoo_ticker"])


def select_standard_in_out(events: pd.DataFrame) -> pd.DataFrame:
    selected = events[
        (events["to_tier"].eq("standard") & events["change_type"].isin(["ADD", "PROMOTION"]))
        | (events["from_tier"].eq("standard") & events["change_type"].isin(["DEL", "DEMOTION"]))
    ].copy()
    selected["event_type"] = np.where(selected["to_tier"].eq("standard"), "inclusion", "deletion")
    selected["strategy_side"] = np.where(selected["event_type"].eq("inclusion"), 1.0, -1.0)
    return selected.sort_values(["event_month", "country", "event_type", "security_name"]).reset_index(drop=True)


def cache_file_name(ticker: str) -> str:
    return str(ticker).replace("/", "_").replace("^", "INDEX_")


def load_price_cache(ticker: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{ticker}.csv"
    if not path.exists():
        path = MULTICOUNTRY_CACHE_DIR / f"{cache_file_name(ticker)}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "Date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"Date": "date"})
    if "Datetime" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"Datetime": "date"})
    # Older cache files may have been written without headers; detect and repair.
    if "date" not in df.columns:
        df = pd.read_csv(
            path,
            header=None,
            names=["date", "ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"],
        )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = ticker
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df["price"] = df[price_col].combine_first(df.get("Close"))
    return df.dropna(subset=["price"]).reset_index(drop=True)


def nearest_trade(prices: pd.DataFrame, target_date: pd.Timestamp, direction: str) -> pd.Series | None:
    if pd.isna(target_date) or prices.empty:
        return None
    if direction == "on_or_after":
        candidates = prices[prices["date"] >= target_date]
        return None if candidates.empty else candidates.iloc[0]
    if direction == "on_or_before":
        candidates = prices[prices["date"] <= target_date]
        return None if candidates.empty else candidates.iloc[-1]
    raise ValueError(f"Unknown direction: {direction}")


def security_return_between(
    prices: pd.DataFrame,
    entry_target: pd.Timestamp,
    exit_target: pd.Timestamp,
) -> dict | None:
    if pd.isna(entry_target) or pd.isna(exit_target) or prices.empty or exit_target < entry_target:
        return None
    entry = nearest_trade(prices, entry_target, "on_or_after")
    exit_ = nearest_trade(prices, exit_target, "on_or_before")
    if entry is None or exit_ is None or exit_["date"] < entry["date"]:
        return None
    entry_px = float(entry["price"])
    exit_px = float(exit_["price"])
    if not np.isfinite(entry_px) or not np.isfinite(exit_px) or entry_px <= 0:
        return None
    return {
        "entry_date": pd.to_datetime(entry["date"]),
        "exit_date": pd.to_datetime(exit_["date"]),
        "entry_px": entry_px,
        "exit_px": exit_px,
        "security_return": exit_px / entry_px - 1.0,
    }


BENCHMARK_CACHE: dict[str, pd.DataFrame] = {}


def load_benchmark_cache(country: str) -> tuple[str | None, pd.DataFrame]:
    benchmark = MARKET_BENCHMARKS.get(str(country))
    if not benchmark:
        return None, pd.DataFrame()
    if benchmark not in BENCHMARK_CACHE:
        BENCHMARK_CACHE[benchmark] = load_price_cache(benchmark)
    return benchmark, BENCHMARK_CACHE[benchmark]


def estimate_beta(
    stock_prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    entry_date: pd.Timestamp,
    lookback: int = 60,
) -> float | None:
    if stock_prices.empty or benchmark_prices.empty or pd.isna(entry_date):
        return None
    cutoff = pd.to_datetime(entry_date) - pd.offsets.BDay(1)
    stock = stock_prices[stock_prices["date"] <= cutoff][["date", "price"]].copy()
    bench = benchmark_prices[benchmark_prices["date"] <= cutoff][["date", "price"]].copy()
    merged = stock.merge(bench, on="date", suffixes=("_stock", "_benchmark")).tail(lookback + 1)
    if len(merged) < 25:
        return None
    returns = merged[["price_stock", "price_benchmark"]].pct_change().dropna()
    if len(returns) < 20:
        return None
    market_var = returns["price_benchmark"].var(ddof=1)
    if pd.isna(market_var) or market_var <= 0:
        return None
    beta = returns["price_stock"].cov(returns["price_benchmark"]) / market_var
    if pd.isna(beta) or not np.isfinite(beta):
        return None
    return float(np.clip(beta, -3.0, 3.0))


def beta_hedge_fields(
    stock_prices: pd.DataFrame,
    country: str,
    side: float,
    entry_date: object,
    exit_date: object,
    security_return: float,
    transaction_cost_bps: float,
) -> dict:
    benchmark, benchmark_prices = load_benchmark_cache(country)
    out = {
        "benchmark_ticker": benchmark,
        "beta_lookback_days": 60,
        "estimated_beta": np.nan,
        "benchmark_return": np.nan,
        "beta_hedged_gross_return": np.nan,
        "beta_hedged_net_return": np.nan,
        "beta_hedge_turnover": np.nan,
        "beta_hedge_round_trip_cost": np.nan,
        "beta_hedge_available": False,
    }
    if benchmark is None or benchmark_prices.empty:
        return out
    entry_ts = pd.to_datetime(entry_date, errors="coerce")
    exit_ts = pd.to_datetime(exit_date, errors="coerce")
    beta = estimate_beta(stock_prices, benchmark_prices, entry_ts)
    benchmark_ret = security_return_between(benchmark_prices, entry_ts, exit_ts)
    if beta is None or benchmark_ret is None:
        return out
    hedge_gross = float(side) * (float(security_return) - beta * float(benchmark_ret["security_return"]))
    hedge_turnover = 2.0 * (1.0 + abs(beta))
    round_trip_cost = hedge_turnover * transaction_cost_bps / 10000.0
    out.update(
        {
            "estimated_beta": beta,
            "benchmark_return": float(benchmark_ret["security_return"]),
            "beta_hedged_gross_return": hedge_gross,
            "beta_hedged_net_return": hedge_gross - round_trip_cost,
            "beta_hedge_turnover": hedge_turnover,
            "beta_hedge_round_trip_cost": round_trip_cost,
            "beta_hedge_available": True,
        }
    )
    return out


def event_window_return(
    prices: pd.DataFrame,
    anchor_date: object,
    entry_offset: int,
    exit_offset: int,
    side: float,
    transaction_cost_bps: float = 10.0,
    country: str | None = None,
) -> dict | None:
    anchor = pd.to_datetime(anchor_date, errors="coerce")
    if pd.isna(anchor):
        return None
    entry_target = anchor + pd.offsets.BDay(entry_offset)
    exit_target = anchor + pd.offsets.BDay(exit_offset)
    if exit_target < entry_target:
        return None

    entry = nearest_trade(prices, entry_target, "on_or_after")
    exit_ = nearest_trade(prices, exit_target, "on_or_before")
    if entry is None or exit_ is None or exit_["date"] < entry["date"]:
        return None

    entry_px = float(entry["price"])
    exit_px = float(exit_["price"])
    if not np.isfinite(entry_px) or not np.isfinite(exit_px) or entry_px <= 0:
        return None

    security_return = exit_px / entry_px - 1.0
    gross_return = float(side) * security_return
    round_trip_cost = 2.0 * transaction_cost_bps / 10000.0
    result = {
        "entry_target_date": entry_target.date().isoformat(),
        "exit_target_date": exit_target.date().isoformat(),
        "entry_date": pd.to_datetime(entry["date"]).date().isoformat(),
        "exit_date": pd.to_datetime(exit_["date"]).date().isoformat(),
        "entry_px": entry_px,
        "exit_px": exit_px,
        "security_return": security_return,
        "gross_return": gross_return,
        "net_return": gross_return - round_trip_cost,
        "transaction_cost_bps": transaction_cost_bps,
        "holding_days": int((pd.to_datetime(exit_["date"]) - pd.to_datetime(entry["date"])).days),
        "turnover": 2.0,
    }
    if country is not None:
        result.update(
            beta_hedge_fields(
                stock_prices=prices,
                country=country,
                side=side,
                entry_date=entry["date"],
                exit_date=exit_["date"],
                security_return=security_return,
                transaction_cost_bps=transaction_cost_bps,
            )
        )
    return result



def load_flow_snapshot_panel() -> pd.DataFrame:
    """Use the existing U.S. event panel to proxy index-flow pressure where available."""
    if not EVENT_WINDOW_PANEL.exists():
        return pd.DataFrame()
    panel = pd.read_csv(EVENT_WINDOW_PANEL)
    needed = {"window_type", "days_from_anchor", "ticker", "effective_date", "Close", "adv_rolling", "market_cap"}
    if missing := needed - set(panel.columns):
        return pd.DataFrame()
    panel = panel[panel["window_type"].eq("effective")].copy()
    panel["days_from_anchor"] = pd.to_numeric(panel["days_from_anchor"], errors="coerce")
    panel = panel[(panel["days_from_anchor"] <= -1) & (panel["days_from_anchor"] >= -10)].copy()
    for col in ["Close", "adv_rolling", "market_cap"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel["effective_date"] = pd.to_datetime(panel["effective_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    panel["dollar_adv"] = panel["Close"] * panel["adv_rolling"]
    panel["expected_flow_proxy"] = panel["market_cap"] / panel["dollar_adv"]
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=["ticker", "effective_date", "expected_flow_proxy"])
    panel = panel.sort_values(["ticker", "effective_date", "days_from_anchor"])
    return panel.groupby(["ticker", "effective_date"], as_index=False).tail(1)[
        ["ticker", "effective_date", "expected_flow_proxy", "market_cap", "dollar_adv"]
    ]


def trailing_price_stats(prices: pd.DataFrame, effective_date: pd.Timestamp, lookback: int = 20) -> dict | None:
    end_target = effective_date - pd.offsets.BDay(1)
    end_row = nearest_trade(prices, end_target, "on_or_before")
    if end_row is None:
        return None
    hist = prices[prices["date"] <= end_row["date"]].tail(lookback + 1).copy()
    if len(hist) < 5:
        return None
    hist["daily_return"] = hist["price"].pct_change()
    dollar_volume = hist["price"] * hist.get("Volume", np.nan)
    return {
        "volatility_20d": float(hist["daily_return"].dropna().std(ddof=1)),
        "dollar_adv20": float(dollar_volume.tail(lookback).mean()),
    }


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return values - values.mean()
    return (values - values.mean()) / std


def neutralize_flow_scores(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    factor_cols = ["pre_effective_runup", "momentum_20d", "volatility_20d", "log_dollar_adv20"]
    features["raw_flow_score"] = np.log1p(features["expected_flow_proxy"].clip(lower=0))
    features["factor_neutral_flow_score"] = np.nan

    def residualize(group: pd.DataFrame) -> pd.Series:
        y = zscore(group["raw_flow_score"])
        x = group[factor_cols].apply(zscore)
        x = x.replace([np.inf, -np.inf], np.nan)
        valid_cols = [c for c in x.columns if x[c].notna().sum() >= 3 and x[c].std(ddof=0) > 0]
        valid = y.notna()
        for col in valid_cols:
            valid &= x[col].notna()
        if valid.sum() <= len(valid_cols) + 1:
            return y - y.mean()
        design = np.column_stack([np.ones(valid.sum()), x.loc[valid, valid_cols].to_numpy(dtype=float)])
        beta = np.linalg.lstsq(design, y.loc[valid].to_numpy(dtype=float), rcond=None)[0]
        resid = pd.Series(index=group.index, dtype=float)
        resid.loc[valid] = y.loc[valid] - design.dot(beta)
        resid.loc[~valid] = y.loc[~valid] - y.loc[valid].mean()
        return resid

    large_month = features.groupby("event_month")["event_id"].transform("count") >= 6
    for month, group in features[large_month].groupby("event_month"):
        features.loc[group.index, "factor_neutral_flow_score"] = residualize(group)
    if features["factor_neutral_flow_score"].isna().any():
        missing_idx = features[features["factor_neutral_flow_score"].isna()].index
        features.loc[missing_idx, "factor_neutral_flow_score"] = residualize(features.loc[missing_idx])

    features["flow_rank_pct"] = features.groupby("event_month")["factor_neutral_flow_score"].rank(pct=True)
    return features


def build_flow_mean_reversion_trades(events: pd.DataFrame, transaction_cost_bps: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inclusion-only post-effective reversal strategy:
    rank inclusions by factor-neutral expected-flow pressure and short the high-flow
    half from effective+1 to effective+5.
    """
    inclusions = events[events["event_type"].eq("inclusion") & events["ticker"].notna()].copy()
    if inclusions.empty:
        return pd.DataFrame(), pd.DataFrame()

    panel = load_flow_snapshot_panel()
    if not panel.empty:
        inclusions["effective_date_key"] = pd.to_datetime(inclusions["effective_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        inclusions = inclusions.merge(
            panel,
            left_on=["ticker", "effective_date_key"],
            right_on=["ticker", "effective_date"],
            how="left",
            suffixes=("", "_panel"),
        )
        if "effective_date_panel" in inclusions.columns:
            inclusions = inclusions.drop(columns=["effective_date_panel"])

    price_cache: dict[str, pd.DataFrame] = {}
    feature_rows: list[dict] = []
    for _, event in inclusions.iterrows():
        ticker = str(event["ticker"])
        if ticker not in price_cache:
            price_cache[ticker] = load_price_cache(ticker)
        prices = price_cache[ticker]
        if prices.empty:
            continue

        announce = pd.to_datetime(event["announce_date"], errors="coerce")
        effective = pd.to_datetime(event["effective_date"], errors="coerce")
        if pd.isna(announce) or pd.isna(effective):
            continue

        pre_effective = security_return_between(prices, announce, effective - pd.offsets.BDay(1))
        momentum = security_return_between(prices, effective - pd.offsets.BDay(20), effective - pd.offsets.BDay(1))
        post = event_window_return(
            prices,
            effective,
            entry_offset=1,
            exit_offset=5,
            side=1.0,
            transaction_cost_bps=transaction_cost_bps,
            country=event["country"],
        )
        stats = trailing_price_stats(prices, effective, lookback=20)
        if pre_effective is None or momentum is None or post is None or stats is None:
            continue

        dollar_adv20 = stats["dollar_adv20"]
        expected_flow_proxy = event.get("expected_flow_proxy")
        if pd.isna(expected_flow_proxy) or not np.isfinite(float(expected_flow_proxy)):
            expected_flow_proxy = 1_000_000_000.0 / dollar_adv20 if dollar_adv20 and dollar_adv20 > 0 else np.nan

        if pd.isna(expected_flow_proxy) or not np.isfinite(float(expected_flow_proxy)) or dollar_adv20 <= 0:
            continue

        feature_rows.append(
            {
                "event_id": event["event_id"],
                "event_month": event["event_month"],
                "country": event["country"],
                "event_type": event["event_type"],
                "change_type": event["change_type"],
                "security_name": event["security_name"],
                "ticker": ticker,
                "announce_date": event["announce_date"],
                "effective_date": event["effective_date"],
                "expected_flow_proxy": float(expected_flow_proxy),
                "pre_effective_runup": float(pre_effective["security_return"]),
                "momentum_20d": float(momentum["security_return"]),
                "volatility_20d": float(stats["volatility_20d"]),
                "dollar_adv20": float(dollar_adv20),
                "log_dollar_adv20": float(np.log(dollar_adv20)),
                "post_security_return": float(post["security_return"]),
                **{f"post_{k}": v for k, v in post.items() if k != "security_return"},
            }
        )

    features = pd.DataFrame(feature_rows)
    if features.empty:
        return pd.DataFrame(), features

    features = neutralize_flow_scores(features)
    selected = features[features["flow_rank_pct"] >= 0.50].copy()
    if selected.empty:
        return pd.DataFrame(), features

    selected["signal_bucket"] = "short_high_flow"
    selected["side"] = -1.0
    round_trip_cost = 2.0 * transaction_cost_bps / 10000.0
    beta_hedge_gross = selected["side"] * (
        selected["post_security_return"] - selected["post_estimated_beta"] * selected["post_benchmark_return"]
    )
    beta_hedge_turnover = 2.0 * (1.0 + selected["post_estimated_beta"].abs())
    beta_hedge_cost = beta_hedge_turnover * transaction_cost_bps / 10000.0
    trades = pd.DataFrame(
        {
            "strategy": "flow_neutral_reversal_short_top50_effective_d+1_to_d+5",
            "anchor": "effective",
            "entry_offset": 1,
            "exit_offset": 5,
            "event_id": selected["event_id"],
            "country": selected["country"],
            "event_type": selected["event_type"],
            "change_type": selected["change_type"],
            "security_name": selected["security_name"],
            "ticker": selected["ticker"],
            "side": selected["side"],
            "signal_bucket": selected["signal_bucket"],
            "announce_date": selected["announce_date"],
            "effective_date": selected["effective_date"],
            "entry_target_date": selected["post_entry_target_date"],
            "exit_target_date": selected["post_exit_target_date"],
            "entry_date": selected["post_entry_date"],
            "exit_date": selected["post_exit_date"],
            "entry_px": selected["post_entry_px"],
            "exit_px": selected["post_exit_px"],
            "security_return": selected["post_security_return"],
            "gross_return": selected["side"] * selected["post_security_return"],
            "net_return": selected["side"] * selected["post_security_return"] - round_trip_cost,
            "transaction_cost_bps": transaction_cost_bps,
            "holding_days": selected["post_holding_days"],
            "turnover": 2.0,
            "benchmark_ticker": selected["post_benchmark_ticker"],
            "beta_lookback_days": selected["post_beta_lookback_days"],
            "estimated_beta": selected["post_estimated_beta"],
            "benchmark_return": selected["post_benchmark_return"],
            "beta_hedged_gross_return": beta_hedge_gross,
            "beta_hedged_net_return": beta_hedge_gross - beta_hedge_cost,
            "beta_hedge_turnover": beta_hedge_turnover,
            "beta_hedge_round_trip_cost": beta_hedge_cost,
            "beta_hedge_available": selected["post_beta_hedge_available"],
            "expected_flow_proxy": selected["expected_flow_proxy"],
            "factor_neutral_flow_score": selected["factor_neutral_flow_score"],
            "flow_rank_pct": selected["flow_rank_pct"],
            "pre_effective_runup": selected["pre_effective_runup"],
            "momentum_20d": selected["momentum_20d"],
            "volatility_20d": selected["volatility_20d"],
            "dollar_adv20": selected["dollar_adv20"],
        }
    )
    return trades.sort_values(["exit_date", "flow_rank_pct"]).reset_index(drop=True), features


def build_buy_hold_trades(events: pd.DataFrame, transaction_cost_bps: float = 10.0) -> pd.DataFrame:
    priced_events = events.dropna(subset=["ticker"]).copy()
    rows: list[dict] = []
    price_cache: dict[str, pd.DataFrame] = {}
    for _, event in priced_events.iterrows():
        ticker = str(event["ticker"])
        if ticker not in price_cache:
            price_cache[ticker] = load_price_cache(ticker)
        prices = price_cache[ticker]
        if prices.empty:
            continue
        # The helper above treats offsets around a single anchor. For buy/hold we need
        # announcement entry and effective-date exit, so calculate it explicitly.
        announce = pd.to_datetime(event["announce_date"], errors="coerce")
        effective = pd.to_datetime(event["effective_date"], errors="coerce")
        if pd.isna(announce) or pd.isna(effective):
            continue
        entry = nearest_trade(prices, announce, "on_or_after")
        exit_ = nearest_trade(prices, effective, "on_or_before")
        if entry is None or exit_ is None or exit_["date"] < entry["date"]:
            continue
        entry_px = float(entry["price"])
        exit_px = float(exit_["price"])
        if not np.isfinite(entry_px) or not np.isfinite(exit_px) or entry_px <= 0:
            continue
        security_return = exit_px / entry_px - 1.0
        gross_return = float(event["strategy_side"]) * security_return
        row = {
            "strategy": "buy_hold_announce_to_effective",
            "anchor": "announce_to_effective",
            "entry_offset": np.nan,
            "exit_offset": np.nan,
            "event_id": event["event_id"],
            "country": event["country"],
            "event_type": event["event_type"],
            "change_type": event["change_type"],
            "security_name": event["security_name"],
            "ticker": ticker,
                "side": event["strategy_side"],
                "announce_date": event["announce_date"],
                "effective_date": event["effective_date"],
                "entry_target_date": announce.date().isoformat(),
                "exit_target_date": effective.date().isoformat(),
                "entry_date": pd.to_datetime(entry["date"]).date().isoformat(),
                "exit_date": pd.to_datetime(exit_["date"]).date().isoformat(),
                "entry_px": entry_px,
                "exit_px": exit_px,
                "security_return": security_return,
                "gross_return": gross_return,
                "net_return": gross_return - 2.0 * transaction_cost_bps / 10000.0,
                "transaction_cost_bps": transaction_cost_bps,
                "holding_days": int((pd.to_datetime(exit_["date"]) - pd.to_datetime(entry["date"])).days),
                "turnover": 2.0,
            }
        row.update(
            beta_hedge_fields(
                stock_prices=prices,
                country=event["country"],
                side=float(event["strategy_side"]),
                entry_date=entry["date"],
                exit_date=exit_["date"],
                security_return=security_return,
                transaction_cost_bps=transaction_cost_bps,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_confirmed_flow_momentum_trades(
    events: pd.DataFrame,
    transaction_cost_bps: float = 10.0,
    top_pct: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inclusions = events[events["event_type"].eq("inclusion") & events["ticker"].notna()].copy()
    rows: list[dict] = []
    price_cache: dict[str, pd.DataFrame] = {}

    for _, event in inclusions.iterrows():
        ticker = str(event["ticker"])
        if ticker not in price_cache:
            price_cache[ticker] = load_price_cache(ticker)
        prices = price_cache[ticker]
        if prices.empty:
            continue

        announce = pd.to_datetime(event["announce_date"], errors="coerce")
        effective = pd.to_datetime(event["effective_date"], errors="coerce")
        if pd.isna(announce) or pd.isna(effective):
            continue

        entry_target = effective + pd.offsets.BDay(-5)
        pre_entry = security_return_between(prices, announce, entry_target - pd.offsets.BDay(1))
        trade = event_window_return(
            prices=prices,
            anchor_date=effective,
            entry_offset=-5,
            exit_offset=-1,
            side=1.0,
            transaction_cost_bps=transaction_cost_bps,
            country=event["country"],
        )
        stats = trailing_price_stats(prices, effective, lookback=20)
        if pre_entry is None or trade is None:
            continue
        event_month = effective.strftime("%Y-%m")
        rows.append(
            {
                "event_id": event["event_id"],
                "event_month": event_month,
                "country": event["country"],
                "event_type": event["event_type"],
                "change_type": event["change_type"],
                "security_name": event["security_name"],
                "ticker": ticker,
                "announce_date": event["announce_date"],
                "effective_date": event["effective_date"],
                "pre_entry_runup": float(pre_entry["security_return"]),
                "dollar_adv20": float(stats["dollar_adv20"]) if stats else np.nan,
                "volatility_20d": float(stats["volatility_20d"]) if stats else np.nan,
                **trade,
            }
        )

    features = pd.DataFrame(rows)
    if features.empty:
        return pd.DataFrame(), features
    features["rank_pct"] = features.groupby(["country", "event_month"])["pre_entry_runup"].rank(pct=True)
    selected = features[features["rank_pct"] >= 1.0 - top_pct].copy()
    selected["strategy"] = f"confirmed_flow_momentum_top{int(top_pct * 100)}_effective_d-5_to_d-1"
    selected["anchor"] = "effective"
    selected["entry_offset"] = -5
    selected["exit_offset"] = -1
    selected["side"] = 1.0
    selected["signal_bucket"] = "top_pre_entry_runup"
    cols = [
        "strategy",
        "anchor",
        "entry_offset",
        "exit_offset",
        "event_id",
        "country",
        "event_type",
        "change_type",
        "security_name",
        "ticker",
        "side",
        "signal_bucket",
        "announce_date",
        "effective_date",
        "entry_target_date",
        "exit_target_date",
        "entry_date",
        "exit_date",
        "entry_px",
        "exit_px",
        "security_return",
        "gross_return",
        "net_return",
        "transaction_cost_bps",
        "holding_days",
        "turnover",
        "benchmark_ticker",
        "beta_lookback_days",
        "estimated_beta",
        "benchmark_return",
        "beta_hedged_gross_return",
        "beta_hedged_net_return",
        "beta_hedge_turnover",
        "beta_hedge_round_trip_cost",
        "beta_hedge_available",
        "pre_entry_runup",
        "rank_pct",
        "dollar_adv20",
        "volatility_20d",
    ]
    return selected[cols].sort_values(["exit_date", "country", "rank_pct"]).reset_index(drop=True), features


def build_low_vol_deletion_trades(
    events: pd.DataFrame,
    transaction_cost_bps: float = 10.0,
    low_vol_rank_cutoff: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Developed simple filter:
    trade only deletion events in the lower-volatility half of their country/month
    deletion basket, using trailing 20-day volatility known at announcement.
    The trade is the historically strongest deletion timing rule: short from
    announcement date -1 to announcement date +1.
    """
    deletions = events[events["event_type"].eq("deletion") & events["ticker"].notna()].copy()
    if deletions.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict] = []
    price_cache: dict[str, pd.DataFrame] = {}
    for _, event in deletions.iterrows():
        ticker = str(event["ticker"])
        if ticker not in price_cache:
            price_cache[ticker] = load_price_cache(ticker)
        prices = price_cache[ticker]
        if prices.empty:
            continue

        announce = pd.to_datetime(event["announce_date"], errors="coerce")
        effective = pd.to_datetime(event["effective_date"], errors="coerce")
        if pd.isna(announce):
            continue

        stats = trailing_price_stats(prices, announce, lookback=20)
        trade = event_window_return(
            prices=prices,
            anchor_date=announce,
            entry_offset=-1,
            exit_offset=1,
            side=float(event["strategy_side"]),
            transaction_cost_bps=transaction_cost_bps,
            country=event["country"],
        )
        if stats is None or trade is None:
            continue

        event_month = effective.strftime("%Y-%m") if pd.notna(effective) else announce.strftime("%Y-%m")
        rows.append(
            {
                "event_id": event["event_id"],
                "event_month": event_month,
                "country": event["country"],
                "event_type": event["event_type"],
                "change_type": event["change_type"],
                "security_name": event["security_name"],
                "ticker": ticker,
                "side": event["strategy_side"],
                "announce_date": event["announce_date"],
                "effective_date": event["effective_date"],
                "volatility_20d": float(stats["volatility_20d"]),
                "dollar_adv20": float(stats["dollar_adv20"]),
                **trade,
            }
        )

    features = pd.DataFrame(rows)
    if features.empty:
        return pd.DataFrame(), features

    group_cols = ["country", "event_month", "event_type"]
    features["basket_size"] = features.groupby(group_cols)["event_id"].transform("count")
    features["low_vol_rank_pct"] = features.groupby(group_cols)["volatility_20d"].rank(pct=True, ascending=False)
    features["liquidity_rank_pct"] = features.groupby(group_cols)["dollar_adv20"].rank(pct=True, ascending=True)
    features["tradable_filter"] = features["dollar_adv20"].gt(0) & features["volatility_20d"].gt(0)
    selected = features[
        features["tradable_filter"] & features["low_vol_rank_pct"].ge(low_vol_rank_cutoff)
    ].copy()
    if selected.empty:
        return pd.DataFrame(), features

    selected["strategy"] = "developed_low_vol_deletion_announce_d-1_to_d+1"
    selected["anchor"] = "announce"
    selected["entry_offset"] = -1
    selected["exit_offset"] = 1
    selected["signal_bucket"] = "lower_volatility_deletion_half"
    cols = [
        "strategy",
        "anchor",
        "entry_offset",
        "exit_offset",
        "event_id",
        "country",
        "event_type",
        "change_type",
        "security_name",
        "ticker",
        "side",
        "signal_bucket",
        "announce_date",
        "effective_date",
        "entry_target_date",
        "exit_target_date",
        "entry_date",
        "exit_date",
        "entry_px",
        "exit_px",
        "security_return",
        "gross_return",
        "net_return",
        "transaction_cost_bps",
        "holding_days",
        "turnover",
        "benchmark_ticker",
        "beta_lookback_days",
        "estimated_beta",
        "benchmark_return",
        "beta_hedged_gross_return",
        "beta_hedged_net_return",
        "beta_hedge_turnover",
        "beta_hedge_round_trip_cost",
        "beta_hedge_available",
        "volatility_20d",
        "dollar_adv20",
        "basket_size",
        "low_vol_rank_pct",
        "liquidity_rank_pct",
    ]
    return selected[cols].sort_values(["exit_date", "country", "low_vol_rank_pct"]).reset_index(drop=True), features


def strategy_rules() -> list[StrategyRule]:
    rules = [
        StrategyRule("announce", -5, 0),
        StrategyRule("announce", -3, 0),
        StrategyRule("announce", -1, 1),
        StrategyRule("announce", 0, 3),
        StrategyRule("announce", 0, 5),
        StrategyRule("effective", -10, -1),
        StrategyRule("effective", -5, -1),
        StrategyRule("effective", -3, 0),
        StrategyRule("effective", -1, 1),
        StrategyRule("effective", 0, 3),
        StrategyRule("effective", 1, 5),
    ]
    return rules


def run_backtests(events: pd.DataFrame, transaction_cost_bps: float = 10.0) -> pd.DataFrame:
    priced_events = events.dropna(subset=["ticker"]).copy()
    rows: list[dict] = []
    price_cache: dict[str, pd.DataFrame] = {}

    for _, event in priced_events.iterrows():
        ticker = str(event["ticker"])
        if ticker not in price_cache:
            price_cache[ticker] = load_price_cache(ticker)
        prices = price_cache[ticker]
        if prices.empty:
            continue

        for rule in strategy_rules():
            anchor_date = event["announce_date"] if rule.anchor == "announce" else event["effective_date"]
            result = event_window_return(
                prices=prices,
                anchor_date=anchor_date,
                entry_offset=rule.entry_offset,
                exit_offset=rule.exit_offset,
                side=float(event["strategy_side"]),
                transaction_cost_bps=transaction_cost_bps,
                country=event["country"],
            )
            if result is None:
                continue
            rows.append(
                {
                    "strategy": rule.label,
                    "anchor": rule.anchor,
                    "entry_offset": rule.entry_offset,
                    "exit_offset": rule.exit_offset,
                    "event_id": event["event_id"],
                    "country": event["country"],
                    "event_type": event["event_type"],
                    "change_type": event["change_type"],
                    "security_name": event["security_name"],
                    "ticker": ticker,
                    "side": event["strategy_side"],
                    "announce_date": event["announce_date"],
                    "effective_date": event["effective_date"],
                    **result,
                }
            )

        # Keep the exact U.S. legacy rule explicit for comparison.
        if pd.notna(event.get("announce_date")) and pd.notna(event.get("effective_date")):
            anchor = pd.to_datetime(event["announce_date"], errors="coerce")
            effective = pd.to_datetime(event["effective_date"], errors="coerce")
            if pd.notna(anchor) and pd.notna(effective):
                entry = nearest_trade(prices, anchor, "on_or_after")
                exit_ = nearest_trade(prices, effective, "on_or_before")
                if entry is not None and exit_ is not None and exit_["date"] >= entry["date"]:
                    entry_px = float(entry["price"])
                    exit_px = float(exit_["price"])
                    security_return = exit_px / entry_px - 1.0
                    gross_return = float(event["strategy_side"]) * security_return
                    row = {
                        "strategy": "announce_to_effective_legacy",
                        "anchor": "announce_to_effective",
                        "entry_offset": np.nan,
                        "exit_offset": np.nan,
                        "event_id": event["event_id"],
                        "country": event["country"],
                        "event_type": event["event_type"],
                        "change_type": event["change_type"],
                        "security_name": event["security_name"],
                        "ticker": ticker,
                        "side": event["strategy_side"],
                        "announce_date": event["announce_date"],
                        "effective_date": event["effective_date"],
                        "entry_target_date": anchor.date().isoformat(),
                        "exit_target_date": effective.date().isoformat(),
                        "entry_date": pd.to_datetime(entry["date"]).date().isoformat(),
                        "exit_date": pd.to_datetime(exit_["date"]).date().isoformat(),
                        "entry_px": entry_px,
                        "exit_px": exit_px,
                        "security_return": security_return,
                        "gross_return": gross_return,
                        "net_return": gross_return - 2.0 * transaction_cost_bps / 10000.0,
                        "transaction_cost_bps": transaction_cost_bps,
                        "holding_days": int((pd.to_datetime(exit_["date"]) - pd.to_datetime(entry["date"])).days),
                        "turnover": 2.0,
                    }
                    row.update(
                        beta_hedge_fields(
                            stock_prices=prices,
                            country=event["country"],
                            side=float(event["strategy_side"]),
                            entry_date=entry["date"],
                            exit_date=exit_["date"],
                            security_return=security_return,
                            transaction_cost_bps=transaction_cost_bps,
                        )
                    )
                    rows.append(row)

    return pd.DataFrame(rows)


def max_drawdown(returns: Iterable[float]) -> float:
    series = pd.Series(list(returns), dtype=float).dropna()
    if series.empty:
        return np.nan
    equity = (1.0 + series).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def trade_daily_path(row: pd.Series, price_cache: dict[str, pd.DataFrame], return_basis: str) -> pd.DataFrame:
    ticker = str(row["ticker"])
    if ticker not in price_cache:
        price_cache[ticker] = load_price_cache(ticker)
    prices = price_cache[ticker]
    if prices.empty:
        return pd.DataFrame()

    entry_date = pd.to_datetime(row["entry_date"], errors="coerce")
    exit_date = pd.to_datetime(row["exit_date"], errors="coerce")
    if pd.isna(entry_date) or pd.isna(exit_date) or exit_date < entry_date:
        return pd.DataFrame()

    path = prices[(prices["date"] >= entry_date) & (prices["date"] <= exit_date)][["date", "price"]].copy()
    if path.empty:
        return pd.DataFrame()
    path = path.sort_values("date")
    path["stock_return"] = path["price"].pct_change().fillna(0.0)

    side = float(row["side"])
    transaction_cost = float(row.get("transaction_cost_bps", 10.0)) / 10000.0
    beta = np.nan
    benchmark_ticker = pd.NA
    benchmark_return = 0.0
    if return_basis == "beta_neutral":
        if not bool(row.get("beta_hedge_available", False)) or pd.isna(row.get("estimated_beta")):
            return pd.DataFrame()
        benchmark_ticker, benchmark_prices = load_benchmark_cache(str(row["country"]))
        if benchmark_ticker is None or benchmark_prices.empty:
            return pd.DataFrame()
        beta = float(row["estimated_beta"])
        bench = benchmark_prices[(benchmark_prices["date"] >= entry_date) & (benchmark_prices["date"] <= exit_date)][
            ["date", "price"]
        ].copy()
        if bench.empty:
            return pd.DataFrame()
        bench = bench.sort_values("date").rename(columns={"price": "benchmark_price"})
        path = path.merge(bench, on="date", how="inner")
        if path.empty:
            return pd.DataFrame()
        path["benchmark_return"] = path["benchmark_price"].pct_change().fillna(0.0)
        benchmark_return = path["benchmark_return"]
        gross = side * (path["stock_return"] - beta * benchmark_return)
        per_side_cost = transaction_cost * (1.0 + abs(beta))
        turnover = 2.0 * (1.0 + abs(beta))
    else:
        gross = side * path["stock_return"]
        per_side_cost = transaction_cost
        turnover = 2.0

    path["daily_gross_return"] = gross
    path["daily_net_return"] = path["daily_gross_return"]
    # Charge one side at entry and one side at exit. If entry == exit, both costs land on that date.
    path.iloc[0, path.columns.get_loc("daily_net_return")] -= per_side_cost
    path.iloc[-1, path.columns.get_loc("daily_net_return")] -= per_side_cost
    path["return_basis"] = return_basis
    path["country"] = row["country"]
    path["event_type"] = row["event_type"]
    path["strategy"] = row["strategy"]
    path["event_id"] = row["event_id"]
    path["ticker"] = ticker
    path["benchmark_ticker"] = benchmark_ticker
    path["estimated_beta"] = beta
    path["turnover"] = turnover
    return path[
        [
            "return_basis",
            "country",
            "event_type",
            "strategy",
            "date",
            "event_id",
            "ticker",
            "benchmark_ticker",
            "estimated_beta",
            "daily_gross_return",
            "daily_net_return",
            "turnover",
        ]
    ]


def build_daily_portfolio_returns(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calendarize event trades into equal-weight daily portfolio returns.

    Event-level Sharpe can look too high for very short windows. This view includes
    zero-return idle business days between the first and last trade in each strategy
    group, which is a more conservative implementation diagnostic.
    """
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()

    price_cache: dict[str, pd.DataFrame] = {}
    daily_rows: list[pd.DataFrame] = []
    for trade_index, row in trades.dropna(subset=["ticker", "entry_date", "exit_date"]).iterrows():
        path = trade_daily_path(row, price_cache, "raw")
        if path.empty:
            continue
        path["trade_key"] = trade_index
        daily_rows.append(path)

    if not daily_rows:
        return pd.DataFrame(), pd.DataFrame()

    trade_daily = pd.concat(daily_rows, ignore_index=True, sort=False)
    group_cols = ["return_basis", "country", "event_type", "strategy"]
    active = (
        trade_daily.groupby(group_cols + ["date"], dropna=False)
        .agg(
            daily_gross_return=("daily_gross_return", "mean"),
            daily_net_return=("daily_net_return", "mean"),
            active_positions=("trade_key", "nunique"),
        )
        .reset_index()
    )
    trade_counts = trade_daily.groupby(group_cols, dropna=False)["trade_key"].nunique().rename("n_trades")
    turnover = trade_daily.drop_duplicates(group_cols + ["trade_key"]).groupby(group_cols, dropna=False)["turnover"].sum()

    filled_groups: list[pd.DataFrame] = []
    for keys, group in active.groupby(group_cols, dropna=False):
        full_dates = pd.date_range(group["date"].min(), group["date"].max(), freq="B")
        filled = pd.DataFrame({"date": full_dates})
        for col, value in zip(group_cols, keys):
            filled[col] = value
        filled = filled.merge(group, on=group_cols + ["date"], how="left")
        filled[["daily_gross_return", "daily_net_return"]] = filled[
            ["daily_gross_return", "daily_net_return"]
        ].fillna(0.0)
        filled["active_positions"] = filled["active_positions"].fillna(0).astype(int)
        filled_groups.append(filled)

    portfolio_daily = pd.concat(filled_groups, ignore_index=True, sort=False)
    portfolio_daily = portfolio_daily.merge(trade_counts.reset_index(), on=group_cols, how="left")
    portfolio_daily = portfolio_daily.merge(turnover.rename("turnover").reset_index(), on=group_cols, how="left")

    summary_rows = []
    for keys, group in portfolio_daily.sort_values("date").groupby(group_cols, dropna=False):
        return_basis, country, event_type, strategy = keys
        net = group["daily_net_return"].astype(float)
        gross = group["daily_gross_return"].astype(float)
        std = net.std(ddof=1)
        sharpe = np.nan if pd.isna(std) or std == 0 else net.mean() / std * np.sqrt(252.0)
        summary_rows.append(
            {
                "return_basis": return_basis,
                "country": country,
                "event_type": event_type,
                "strategy": strategy,
                "n_trades": int(group["n_trades"].iloc[0]),
                "calendar_days": int(len(group)),
                "active_days": int((group["active_positions"] > 0).sum()),
                "average_active_positions": float(group["active_positions"].mean()),
                "cumulative_gross_return": float((1.0 + gross).prod() - 1.0),
                "cumulative_net_return": float((1.0 + net).prod() - 1.0),
                "average_daily_net_return": float(net.mean()),
                "volatility_daily_net": float(std) if pd.notna(std) else np.nan,
                "sharpe_daily_net": float(sharpe) if pd.notna(sharpe) else np.nan,
                "hit_rate_daily_net": float((net > 0).mean()),
                "max_drawdown_net": max_drawdown(net),
                "turnover": float(group["turnover"].iloc[0]),
            }
        )
    portfolio_summary = pd.DataFrame(summary_rows)
    return (
        portfolio_daily.sort_values(group_cols + ["date"]).reset_index(drop=True),
        portfolio_summary.sort_values(["sharpe_daily_net", "cumulative_net_return"], ascending=False).reset_index(drop=True),
    )


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["country", "event_type", "strategy"]
    for keys, group in trades.sort_values("exit_date").groupby(group_cols, dropna=False):
        country, event_type, strategy = keys
        gross = group["gross_return"].astype(float)
        net = group["net_return"].astype(float)
        std = net.std(ddof=1)
        avg_holding = group["holding_days"].mean()
        event_sharpe = np.nan if not np.isfinite(std) or std == 0 else net.mean() / std
        annualized_sharpe = (
            np.nan
            if pd.isna(event_sharpe) or pd.isna(avg_holding) or avg_holding <= 0
            else event_sharpe * np.sqrt(252.0 / avg_holding)
        )
        rows.append(
            {
                "country": country,
                "event_type": event_type,
                "strategy": strategy,
                "n_trades": int(len(group)),
                "cumulative_gross_return": float((1.0 + gross).prod() - 1.0),
                "cumulative_net_return": float((1.0 + net).prod() - 1.0),
                "average_event_gross_return": float(gross.mean()),
                "average_event_net_return": float(net.mean()),
                "hit_rate_net": float((net > 0).mean()),
                "volatility_event_net": float(std) if pd.notna(std) else np.nan,
                "sharpe_event_net": float(event_sharpe) if pd.notna(event_sharpe) else np.nan,
                "sharpe_annualized_net": float(annualized_sharpe) if pd.notna(annualized_sharpe) else np.nan,
                "max_drawdown_net": max_drawdown(net),
                "average_holding_days": float(avg_holding),
                "turnover": float(group["turnover"].sum()),
                "transaction_cost_bps": float(group["transaction_cost_bps"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.sort_values(["sharpe_annualized_net", "average_event_net_return"], ascending=False).reset_index(drop=True)


def summarize_beta_neutral_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "beta_hedged_net_return" not in trades.columns:
        return pd.DataFrame()
    hedged = trades[trades["beta_hedge_available"].eq(True)].dropna(subset=["beta_hedged_net_return"]).copy()
    if hedged.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["country", "event_type", "strategy"]
    for keys, group in hedged.sort_values("exit_date").groupby(group_cols, dropna=False):
        country, event_type, strategy = keys
        net = group["beta_hedged_net_return"].astype(float)
        gross = group["beta_hedged_gross_return"].astype(float)
        std = net.std(ddof=1)
        avg_holding = group["holding_days"].mean()
        event_sharpe = np.nan if not np.isfinite(std) or std == 0 else net.mean() / std
        annualized_sharpe = (
            np.nan
            if pd.isna(event_sharpe) or pd.isna(avg_holding) or avg_holding <= 0
            else event_sharpe * np.sqrt(252.0 / avg_holding)
        )
        hedge_turnover = (
            group["beta_hedge_turnover"].astype(float)
            if "beta_hedge_turnover" in group.columns
            else group["turnover"].astype(float)
        )
        rows.append(
            {
                "country": country,
                "event_type": event_type,
                "strategy": strategy,
                "return_basis": "beta_neutral",
                "n_trades": int(len(group)),
                "cumulative_gross_return": float((1.0 + gross).prod() - 1.0),
                "cumulative_net_return": float((1.0 + net).prod() - 1.0),
                "average_event_gross_return": float(gross.mean()),
                "average_event_net_return": float(net.mean()),
                "hit_rate_net": float((net > 0).mean()),
                "volatility_event_net": float(std) if pd.notna(std) else np.nan,
                "sharpe_event_net": float(event_sharpe) if pd.notna(event_sharpe) else np.nan,
                "sharpe_annualized_net": float(annualized_sharpe) if pd.notna(annualized_sharpe) else np.nan,
                "max_drawdown_net": max_drawdown(net),
                "average_holding_days": float(avg_holding),
                "turnover": float(hedge_turnover.sum()),
                "transaction_cost_bps": float(group["transaction_cost_bps"].mean()),
                "average_beta": float(group["estimated_beta"].astype(float).mean()),
                "average_beta_hedge_turnover": float(hedge_turnover.mean()),
                "benchmark_ticker": ";".join(sorted(group["benchmark_ticker"].dropna().astype(str).unique())),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.sort_values(["sharpe_annualized_net", "average_event_net_return"], ascending=False).reset_index(drop=True)


def coverage_summary(events: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    priced_ids = set(trades["event_id"].unique()) if not trades.empty else set()
    rows = []
    for (country, event_type), group in events.groupby(["country", "event_type"], dropna=False):
        rows.append(
            {
                "country": country,
                "event_type": event_type,
                "standard_events": int(len(group)),
                "events_with_ticker": int(group["ticker"].notna().sum()) if "ticker" in group else 0,
                "events_with_cached_price": int(group["event_id"].isin(priced_ids).sum()),
                "missing_ticker": int(group["ticker"].isna().sum()) if "ticker" in group else int(len(group)),
                "missing_cached_price_after_ticker": int(
                    group["ticker"].notna().sum() - group["event_id"].isin(priced_ids).sum()
                )
                if "ticker" in group
                else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["country", "event_type"]).reset_index(drop=True)


def write_yahoo_ticker_diagnostics(events: pd.DataFrame) -> None:
    if not MULTICOUNTRY_MAPPING_PATH.exists():
        return
    mapping = pd.read_csv(MULTICOUNTRY_MAPPING_PATH)
    diag = events[["country", "security_name", "event_type", "ticker"]].drop_duplicates()
    diag = diag.merge(mapping, on=["country", "security_name"], how="left", suffixes=("_event", "_mapping"))
    if "ticker" not in diag.columns:
        diag["ticker"] = diag.get("ticker_event").combine_first(diag.get("ticker_mapping"))
    if (MULTICOUNTRY_YAHOO_DIR / "yahoo_price_fetch_status.csv").exists():
        fetch = pd.read_csv(MULTICOUNTRY_YAHOO_DIR / "yahoo_price_fetch_status.csv")
        fetch = fetch.rename(columns={"status": "price_fetch_status", "reason": "price_fetch_reason"})
        diag = diag.merge(fetch[["ticker", "price_fetch_status", "rows", "price_fetch_reason"]], on="ticker", how="left")
    else:
        diag["price_fetch_status"] = pd.NA
        diag["rows"] = pd.NA
        diag["price_fetch_reason"] = pd.NA

    def diagnose(row: pd.Series) -> str:
        status = row.get("status")
        fetch_status = row.get("price_fetch_status")
        if pd.isna(status):
            return "not_in_yahoo_mapping_file"
        if status in {"no_country_candidate", "no_search_result"}:
            return "name_search_unmapped"
        if status == "unsupported_country":
            return "unsupported_country"
        if pd.isna(row.get("ticker")):
            return "missing_ticker"
        if fetch_status == "no_price_data":
            return "mapped_but_yahoo_has_no_event_window_history"
        if fetch_status in {"downloaded", "cached"}:
            return "mapped_and_priced"
        return "mapped_price_status_unknown"

    diag["diagnosis"] = diag.apply(diagnose, axis=1)
    diag.to_csv(OUT_DIR / "yahoo_ticker_diagnostics.csv", index=False)


def basic_vs_flow_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    flow = summary[summary["strategy"].str.startswith("flow_neutral_reversal", na=False)].copy()
    basic_pool = summary[
        ~summary["strategy"].str.startswith("flow_neutral_reversal", na=False)
        & summary["country"].eq("USA")
        & summary["event_type"].eq("inclusion")
    ].copy()
    if flow.empty or basic_pool.empty:
        return pd.DataFrame()
    basic = basic_pool.sort_values("sharpe_annualized_net", ascending=False).head(1).copy()
    flow = flow.sort_values("sharpe_annualized_net", ascending=False).head(1).copy()
    basic["strategy_family"] = "basic_event_timing"
    flow["strategy_family"] = "flow_neutral_mean_reversion"
    cols = [
        "strategy_family",
        "country",
        "event_type",
        "strategy",
        "n_trades",
        "cumulative_net_return",
        "average_event_net_return",
        "hit_rate_net",
        "volatility_event_net",
        "sharpe_annualized_net",
        "max_drawdown_net",
        "average_holding_days",
        "turnover",
    ]
    return pd.concat([basic, flow], ignore_index=True)[cols]


def buy_hold_vs_best_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    candidates = []

    buy_hold = summary[summary["strategy"].eq("buy_hold_announce_to_effective")].copy()
    if not buy_hold.empty:
        row = buy_hold.sort_values("sharpe_annualized_net", ascending=False).head(1).copy()
        row["strategy_family"] = "buy_hold"
        candidates.append(row)

    timing = summary[
        ~summary["strategy"].str.startswith("confirmed_flow_momentum", na=False)
        & ~summary["strategy"].str.startswith("flow_neutral_reversal", na=False)
        & ~summary["strategy"].str.startswith("developed_low_vol_deletion", na=False)
        & ~summary["strategy"].eq("buy_hold_announce_to_effective")
    ].copy()
    if not timing.empty:
        row = timing.sort_values("sharpe_annualized_net", ascending=False).head(1).copy()
        row["strategy_family"] = "best_simple_timing"
        candidates.append(row)

    confirmed = summary[summary["strategy"].str.startswith("confirmed_flow_momentum", na=False)].copy()
    if not confirmed.empty:
        row = confirmed.sort_values("sharpe_annualized_net", ascending=False).head(1).copy()
        row["strategy_family"] = "confirmed_flow_momentum"
        candidates.append(row)

    developed = summary[summary["strategy"].str.startswith("developed_low_vol_deletion", na=False)].copy()
    if not developed.empty:
        row = developed.sort_values("sharpe_annualized_net", ascending=False).head(1).copy()
        row["strategy_family"] = "developed_low_vol_deletion"
        candidates.append(row)

    if not candidates:
        return pd.DataFrame()
    cols = [
        "strategy_family",
        "country",
        "event_type",
        "strategy",
        "n_trades",
        "cumulative_net_return",
        "average_event_net_return",
        "hit_rate_net",
        "volatility_event_net",
        "sharpe_annualized_net",
        "max_drawdown_net",
        "average_holding_days",
        "turnover",
    ]
    return pd.concat(candidates, ignore_index=True)[cols]


def strategy_family(strategy: object) -> str:
    name = clean_text(strategy)
    if name == "buy_hold_announce_to_effective":
        return "buy_hold"
    if name == "announce_to_effective_legacy":
        return "legacy_buy_hold"
    if name.startswith("developed_low_vol_deletion"):
        return "developed_low_vol_deletion"
    if name.startswith("confirmed_flow_momentum"):
        return "confirmed_flow_momentum"
    if name.startswith("flow_neutral_reversal"):
        return "flow_mean_reversion"
    return "event_timing"


def build_market_event_playbook(portfolio_summary: pd.DataFrame) -> pd.DataFrame:
    if portfolio_summary.empty:
        return pd.DataFrame()

    raw = portfolio_summary[portfolio_summary["return_basis"].eq("raw")].copy()
    if raw.empty:
        return pd.DataFrame()
    raw["strategy_family"] = raw["strategy"].map(strategy_family)

    rows = []
    for (country, event_type), group in raw.groupby(["country", "event_type"], dropna=False):
        details = MARKET_DETAILS.get(country, {})
        buy_hold = group[group["strategy"].eq("buy_hold_announce_to_effective")].copy()
        buy_hold = buy_hold.sort_values("sharpe_daily_net", ascending=False).head(1)

        candidates = group[
            ~group["strategy_family"].isin(["buy_hold", "legacy_buy_hold"])
        ].copy()
        candidates = candidates[candidates["n_trades"] >= 5]
        best = candidates.sort_values(["sharpe_daily_net", "cumulative_net_return"], ascending=False).head(1)

        developed = group[group["strategy_family"].eq("developed_low_vol_deletion")].copy()
        developed = developed.sort_values("sharpe_daily_net", ascending=False).head(1)

        row = {
            "country": country,
            "event_type": event_type,
            "stock_market": details.get("stock_market", country),
            "primary_exchanges": details.get("primary_exchanges", ""),
            "yahoo_suffixes": details.get("yahoo_suffixes", ""),
            "benchmark_ticker": details.get("benchmark_ticker", MARKET_BENCHMARKS.get(country, "")),
        }

        if not buy_hold.empty:
            bh = buy_hold.iloc[0]
            row.update(
                {
                    "buy_hold_strategy": bh["strategy"],
                    "buy_hold_n_trades": int(bh["n_trades"]),
                    "buy_hold_daily_sharpe": float(bh["sharpe_daily_net"]),
                    "buy_hold_cumulative_net_return": float(bh["cumulative_net_return"]),
                    "buy_hold_max_drawdown": float(bh["max_drawdown_net"]),
                    "buy_hold_active_days": int(bh["active_days"]),
                }
            )

        if not best.empty:
            b = best.iloc[0]
            row.update(
                {
                    "best_active_strategy_family": b["strategy_family"],
                    "best_active_strategy": b["strategy"],
                    "best_active_n_trades": int(b["n_trades"]),
                    "best_active_daily_sharpe": float(b["sharpe_daily_net"]),
                    "best_active_cumulative_net_return": float(b["cumulative_net_return"]),
                    "best_active_max_drawdown": float(b["max_drawdown_net"]),
                    "best_active_active_days": int(b["active_days"]),
                }
            )

        if not developed.empty:
            d = developed.iloc[0]
            row.update(
                {
                    "developed_strategy": d["strategy"],
                    "developed_n_trades": int(d["n_trades"]),
                    "developed_daily_sharpe": float(d["sharpe_daily_net"]),
                    "developed_cumulative_net_return": float(d["cumulative_net_return"]),
                    "developed_max_drawdown": float(d["max_drawdown_net"]),
                }
            )

        bh_sharpe = row.get("buy_hold_daily_sharpe", np.nan)
        best_sharpe = row.get("best_active_daily_sharpe", np.nan)
        if pd.notna(bh_sharpe) and pd.notna(best_sharpe):
            row["best_minus_buy_hold_daily_sharpe"] = best_sharpe - bh_sharpe
            row["best_minus_buy_hold_cumulative_return"] = row.get("best_active_cumulative_net_return", np.nan) - row.get(
                "buy_hold_cumulative_net_return", np.nan
            )
            if best_sharpe > bh_sharpe:
                row["recommendation"] = row.get("best_active_strategy", "")
            else:
                row["recommendation"] = "buy_hold_announce_to_effective"
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = ["India", "Indonesia", "Korea", "USA"]
    out["country_order"] = out["country"].map({country: i for i, country in enumerate(order)}).fillna(99)
    out["event_order"] = out["event_type"].map({"inclusion": 0, "deletion": 1}).fillna(99)
    out = out.sort_values(["country_order", "event_order"]).drop(columns=["country_order", "event_order"])
    return out.reset_index(drop=True)


def save_buy_hold_vs_best_chart(
    summary: pd.DataFrame,
    trades: pd.DataFrame,
    return_col: str = "net_return",
    comparison_name: str = "buy_hold_vs_best_strategy_comparison.csv",
    chart_name: str = "buy_hold_vs_best_strategy.png",
    title_suffix: str = "",
) -> None:
    comp = buy_hold_vs_best_comparison(summary)
    if comp.empty or trades.empty:
        return
    if return_col not in trades.columns:
        return
    comp.to_csv(OUT_DIR / comparison_name, index=False)
    colors = {
        "buy_hold": "#6f6f6f",
        "best_simple_timing": "#1b4d89",
        "confirmed_flow_momentum": "#2f6f73",
        "developed_low_vol_deletion": "#b45f06",
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    labels = comp["strategy_family"].str.replace("_", "\n").tolist()
    x = np.arange(len(comp))

    for _, row in comp.iterrows():
        strat_trades = trades[
            trades["country"].eq(row["country"])
            & trades["event_type"].eq(row["event_type"])
            & trades["strategy"].eq(row["strategy"])
        ].dropna(subset=[return_col]).sort_values("exit_date")
        if strat_trades.empty:
            continue
        equity = (1.0 + strat_trades[return_col].astype(float)).cumprod()
        axes[0, 0].plot(
            pd.to_datetime(strat_trades["exit_date"]),
            equity,
            label=row["strategy_family"],
            color=colors.get(row["strategy_family"], "#333333"),
            linewidth=2,
        )

    axes[0, 0].set_title(f"Net Equity Curve{title_suffix}")
    axes[0, 0].set_ylabel("Cumulative net equity")
    axes[0, 0].legend()

    bar_colors = [colors.get(v, "#333333") for v in comp["strategy_family"]]
    axes[0, 1].bar(x, comp["sharpe_annualized_net"], color=bar_colors)
    axes[0, 1].set_title("Annualized Net Sharpe")
    axes[0, 1].set_xticks(x, labels)

    axes[1, 0].bar(x, comp["average_event_net_return"], color=bar_colors)
    axes[1, 0].set_title("Average Net Event Return")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    axes[1, 1].bar(x, comp["max_drawdown_net"], color=bar_colors)
    axes[1, 1].set_title("Max Drawdown")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    fig.suptitle(f"Buy/Hold vs Event Strategy Families{title_suffix}", fontsize=14)
    fig.tight_layout()
    fig.savefig(CHART_DIR / chart_name, dpi=180)
    plt.close(fig)


def save_portfolio_comparison_chart(summary: pd.DataFrame, portfolio_summary: pd.DataFrame) -> None:
    if summary.empty or portfolio_summary.empty:
        return

    comp = buy_hold_vs_best_comparison(summary)
    if comp.empty:
        return

    comp = comp.copy()
    comp["return_basis"] = "raw"
    comp = comp.merge(
        portfolio_summary,
        on=["return_basis", "country", "event_type", "strategy"],
        how="left",
        suffixes=("_event", "_daily"),
    )
    comp = comp.dropna(subset=["sharpe_daily_net"])
    if comp.empty:
        return

    comp.to_csv(OUT_DIR / "buy_hold_vs_best_portfolio_daily_comparison.csv", index=False)

    colors = {
        "buy_hold": "#6f6f6f",
        "best_simple_timing": "#1b4d89",
        "confirmed_flow_momentum": "#2f6f73",
        "developed_low_vol_deletion": "#b45f06",
    }
    labels = comp["strategy_family"].str.replace("_", "\n").tolist()
    x = np.arange(len(comp))
    bar_colors = [colors.get(v, "#333333") for v in comp["strategy_family"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0, 0].bar(x, comp["sharpe_daily_net"], color=bar_colors)
    axes[0, 1].bar(x, comp["cumulative_net_return_daily"], color=bar_colors)
    axes[1, 0].bar(x, comp["max_drawdown_net_daily"], color=bar_colors)
    axes[1, 1].bar(x, comp["active_days"], color=bar_colors)

    axes[0, 0].set_title("Calendar Daily Sharpe")
    axes[0, 1].set_title("Cumulative Net Return")
    axes[0, 1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1, 0].set_title("Max Drawdown")
    axes[1, 0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1, 1].set_title("Active Trading Days")

    for ax in axes.ravel():
        ax.set_xticks(x, labels)

    fig.suptitle("Executable Daily Portfolio View: Buy/Hold vs Event Strategies", fontsize=14)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "buy_hold_vs_best_portfolio_daily_comparison.png", dpi=180)
    plt.close(fig)


def save_market_event_playbook_chart(playbook: pd.DataFrame) -> None:
    if playbook.empty:
        return
    needed = {"buy_hold_daily_sharpe", "best_active_daily_sharpe"}
    if missing := needed - set(playbook.columns):
        return

    chart = playbook.dropna(subset=["buy_hold_daily_sharpe", "best_active_daily_sharpe"]).copy()
    if chart.empty:
        return
    chart["label"] = chart["country"] + "\n" + chart["event_type"]
    x = np.arange(len(chart))
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].bar(x - width / 2, chart["buy_hold_daily_sharpe"], width=width, label="buy/hold", color="#6f6f6f")
    axes[0].bar(x + width / 2, chart["best_active_daily_sharpe"], width=width, label="best active rule", color="#1b4d89")
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_title("Daily Sharpe by Market and MSCI Event Type")
    axes[0].set_ylabel("Calendarized daily Sharpe")
    axes[0].set_xticks(x, chart["label"])
    axes[0].legend()

    deltas = chart["best_minus_buy_hold_daily_sharpe"].fillna(0.0)
    colors = np.where(deltas >= 0, "#2f6f73", "#b45f06")
    axes[1].bar(x, deltas, color=colors)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_title("Best Active Rule Minus Buy/Hold")
    axes[1].set_ylabel("Sharpe delta")
    axes[1].set_xticks(x, chart["label"])

    fig.tight_layout()
    fig.savefig(CHART_DIR / "market_event_buy_hold_vs_best_rule.png", dpi=180)
    plt.close(fig)


def save_basic_vs_flow_chart(summary: pd.DataFrame, trades: pd.DataFrame) -> None:
    comp = basic_vs_flow_comparison(summary)
    if comp.empty or trades.empty:
        return
    comp.to_csv(OUT_DIR / "basic_vs_flow_strategy_comparison.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    colors = {
        "basic_event_timing": "#1b4d89",
        "flow_neutral_mean_reversion": "#b45f06",
    }

    for _, row in comp.iterrows():
        strat_trades = trades[
            trades["country"].eq(row["country"])
            & trades["event_type"].eq(row["event_type"])
            & trades["strategy"].eq(row["strategy"])
        ].sort_values("exit_date")
        if strat_trades.empty:
            continue
        equity = (1.0 + strat_trades["net_return"].astype(float)).cumprod()
        axes[0, 0].plot(
            pd.to_datetime(strat_trades["exit_date"]),
            equity,
            label=row["strategy_family"],
            color=colors.get(row["strategy_family"], "#333333"),
            linewidth=2,
        )

    axes[0, 0].set_title("Net Equity Curve")
    axes[0, 0].set_ylabel("Cumulative net equity")
    axes[0, 0].legend()

    labels = comp["strategy_family"].str.replace("_", "\n").tolist()
    x = np.arange(len(comp))
    axes[0, 1].bar(x, comp["sharpe_annualized_net"], color=[colors.get(v, "#333333") for v in comp["strategy_family"]])
    axes[0, 1].set_title("Annualized Net Sharpe")
    axes[0, 1].set_xticks(x, labels)

    axes[1, 0].bar(x, comp["average_event_net_return"], color=[colors.get(v, "#333333") for v in comp["strategy_family"]])
    axes[1, 0].set_title("Average Net Event Return")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    axes[1, 1].bar(x, comp["max_drawdown_net"], color=[colors.get(v, "#333333") for v in comp["strategy_family"]])
    axes[1, 1].set_title("Max Drawdown")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    fig.suptitle("Basic Event Timing vs Flow-Neutral Post-Inclusion Mean Reversion", fontsize=14)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "basic_vs_flow_mean_reversion.png", dpi=180)
    plt.close(fig)


def save_charts(
    summary: pd.DataFrame,
    trades: pd.DataFrame,
    beta_summary: pd.DataFrame | None = None,
    portfolio_summary: pd.DataFrame | None = None,
) -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return

    top = summary.sort_values("sharpe_annualized_net", ascending=False).head(20).copy()
    top["label"] = top["country"] + " " + top["event_type"] + "\n" + top["strategy"]
    plt.figure(figsize=(12, 7))
    plt.barh(top["label"][::-1], top["sharpe_annualized_net"][::-1], color="#2f6f73")
    plt.xlabel("Annualized Sharpe, net of costs")
    plt.title("Top MSCI Rebalance Event Timing Rules")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "top_strategy_sharpe.png", dpi=180)
    plt.close()

    pivot = summary.pivot_table(
        index=["country", "event_type"],
        columns="strategy",
        values="average_event_net_return",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(max(16, 0.85 * len(pivot.columns)), max(4, 0.55 * len(pivot))))
    image = ax.imshow(pivot.fillna(0.0).values, aspect="auto", cmap="RdYlGn")
    fig.colorbar(image, ax=ax, label="Average event net return")
    ax.set_yticks(range(len(pivot.index)), [f"{a} {b}" for a, b in pivot.index])
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=70, ha="right")
    ax.set_title("Average Net Return by Country, Event Type, and Timing Rule")
    fig.subplots_adjust(left=0.16, right=0.94, top=0.88, bottom=0.46)
    fig.savefig(CHART_DIR / "strategy_return_heatmap.png", dpi=180)
    plt.close(fig)

    if not trades.empty:
        best_key = summary.iloc[0][["country", "event_type", "strategy"]].to_dict()
        best = trades[
            trades["country"].eq(best_key["country"])
            & trades["event_type"].eq(best_key["event_type"])
            & trades["strategy"].eq(best_key["strategy"])
        ].sort_values("exit_date")
        equity = (1.0 + best["net_return"].astype(float)).cumprod()
        plt.figure(figsize=(10, 5))
        plt.plot(pd.to_datetime(best["exit_date"]), equity, color="#1b4d89", linewidth=2)
        plt.ylabel("Cumulative net equity")
        plt.title(f"Best Rule Equity: {best_key['country']} {best_key['event_type']} {best_key['strategy']}")
        plt.tight_layout()
        plt.savefig(CHART_DIR / "best_strategy_equity_curve.png", dpi=180)
        plt.close()

    save_basic_vs_flow_chart(summary, trades)
    save_buy_hold_vs_best_chart(summary, trades)
    if portfolio_summary is not None:
        save_portfolio_comparison_chart(summary, portfolio_summary)


def write_methodology(
    events: pd.DataFrame,
    coverage: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_line = "No priced strategy results were available."
    if not summary.empty:
        best = summary.iloc[0]
        best_line = (
            f"Best priced combination: {best['country']} {best['event_type']} "
            f"{best['strategy']} with annualized net Sharpe {best['sharpe_annualized_net']:.2f}, "
            f"average net event return {best['average_event_net_return']:.2%}, "
            f"and {int(best['n_trades'])} trades."
        )
    country_counts = events.groupby(["country", "event_type"]).size().reset_index(name="events")
    coverage_text = coverage.to_string(index=False) if not coverage.empty else "No standard in/out events."
    count_text = country_counts.to_string(index=False) if not country_counts.empty else "No standard in/out events."
    market_text = pd.DataFrame(
        [
            {"country": country, **details}
            for country, details in MARKET_DETAILS.items()
        ]
    ).sort_values("country").to_string(index=False)

    text = f"""# MSCI Multi-Country Rebalance Strategy Pipeline

## Existing U.S. Logic Replicated

The original U.S. workflow extracts MSCI public-list additions and deletions, assigns each leg to
standard, small_cap, or micro_cap based on source file type, reconciles paired legs into ADD, DEL,
PROMOTION, or DEMOTION events, filters to Standard index inclusions/deletions, parses Yahoo tickers
from MSCI security ticker strings, joins cached daily price/volume data, and evaluates event-window
returns around announcement and effective dates.

This pipeline generalizes that flow for Indonesia, Korea, India, and the U.S. It standardizes country,
security name, tier, event type, announcement date, effective date, ticker, event side, and output
schemas. U.S. rows reuse the existing OCR and ticker-enriched data when available; non-U.S. rows are
parsed from the same MSCI public-list PDFs via word-coordinate splitting of the Additions/Deletions
columns.

## Market Universe

{market_text}

## Assumptions

- Standard inclusions are events where `to_tier == standard`; Standard deletions are events where
  `from_tier == standard`.
- Inclusion strategies are long the affected security; deletion strategies are short the affected
  security.
- The flow-neutral mean-reversion strategy uses Standard inclusions only. It proxies expected flow
  pressure as market-cap divided by dollar ADV from the cached U.S. event panel when available,
  falling back to inverse 20-day dollar ADV. It residualizes that score against pre-effective run-up,
  20-day momentum, 20-day volatility, and log dollar ADV, then shorts the high-flow half from
  effective date +1 to +5.
- Transaction costs are modeled as a round trip cost of 2 x `transaction_cost_bps`, defaulting to
  10 bps per side.
- Entry dates use the first cached trading day on or after the target date. Exit dates use the last
  cached trading day on or before the target date.
- Non-U.S. MSCI public lists in this repository generally do not include exchange ticker strings, and
  therefore require Yahoo name-search mapping. The mapping diagnostics are saved separately so
  missing search hits, invalid/delisted tickers, and missing price history are not mixed together.
- Average daily volume and dollar ADV come from cached Yahoo daily price/volume files. The flow proxy
  is a tradability/flow-pressure proxy, not official MSCI passive AUM flow.
- `buy_hold_announce_to_effective` is the simple benchmark: enter on/after announcement and exit
  on/before effective date, using long inclusions and short deletions.
- `confirmed_flow_momentum_top20_effective_d-5_to_d-1` ranks inclusions within each country and
  rebalance month by run-up already visible before the `effective -5` entry date, then buys the top
  20% from `effective -5` to `effective -1`.
- `developed_low_vol_deletion_announce_d-1_to_d+1` is the custom developed strategy. It shorts
  Standard deletions from announcement -1 to +1 only when the stock is in the lower-volatility half
  of its country/month deletion basket, using trailing 20-day volatility and Yahoo 20-day traded
  value known at announcement.
- The annualized Sharpe columns are event-window diagnostics, not production portfolio Sharpe ratios.
  Very short holding periods and small samples can produce unstable values such as double-digit
  annualized Sharpe. Use `sharpe_event_net`, `n_trades`, drawdown, coverage, borrow/short feasibility,
  and the daily portfolio comparison before treating a rule as executable.
- `portfolio_daily_summary.csv` is the more conservative implementation view. It calendarizes the
  event trades into equal-weight daily strategy portfolios, fills idle business days with zero return,
  and reports daily Sharpe, daily volatility, active days, drawdown, and cumulative return.

## Standard Event Counts

{count_text}

## Price Coverage

{coverage_text}

## Best Timing Result

{best_line}

## Outputs

- `raw_event_legs_all_countries.csv`: standardized raw addition/deletion legs.
- `cleaned_events_all_countries.csv`: reconciled all-tier events.
- `standard_in_out_events_all_countries.csv`: unified Standard inclusion/deletion event set.
- `event_window_trade_results.csv`: event-level backtest trades for priced events.
- `portfolio_daily_returns.csv`: calendarized raw daily portfolio returns. This file is ignored by
  Git by default because it can be large.
- `portfolio_daily_summary.csv`: daily portfolio metrics designed to sanity-check executable Sharpe.
- `market_event_rebalance_playbook.csv`: country-by-country and inclusion/deletion playbook comparing
  each market/event type's buy-hold benchmark with the best active rebalance rule.
- `buy_hold_trades.csv`: simple buy/hold benchmark trades.
- `confirmed_flow_momentum_features.csv`: pre-entry run-up and ADV features used by the confirmed-flow strategy.
- `confirmed_flow_momentum_trades.csv`: selected confirmed-flow momentum trades.
- `flow_mean_reversion_features.csv`: inclusion-level expected-flow and neutralized-factor signals.
- `flow_mean_reversion_trades.csv`: selected high-flow short reversal trades.
- `developed_low_vol_deletion_features.csv`: pre-announcement volatility and ADV features used by
  the low-volatility deletion strategy.
- `developed_low_vol_deletion_trades.csv`: selected low-volatility deletion trades.
- `basic_vs_flow_strategy_comparison.csv`: side-by-side metrics for the best basic timing rule and
  the flow-neutral mean-reversion rule.
- `buy_hold_vs_best_strategy_comparison.csv`: side-by-side metrics for buy/hold, best timing, and
  strategy families, including the low-volatility deletion filter.
- `buy_hold_vs_best_portfolio_daily_comparison.csv`: raw daily portfolio view for buy/hold and event
  strategy families.
- `strategy_summary.csv`: performance metrics by country, event type, and timing rule.
- `price_coverage_summary.csv`: missing ticker/price diagnostics by country and event type.
- `charts/`: Sharpe ranking, return heatmap, best-rule equity curve, and raw side-by-side comparison
  charts.
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def run_pipeline(transaction_cost_bps: float = 10.0, use_existing_usa: bool = True) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    legs = build_raw_legs(use_existing_usa=use_existing_usa)
    events = reconcile_legs(legs)
    events = attach_existing_usa_tickers(events)
    events = attach_multicountry_yahoo_tickers(events)
    standard_events = select_standard_in_out(events)

    timing_trades = run_backtests(standard_events, transaction_cost_bps=transaction_cost_bps)
    buy_hold_trades = build_buy_hold_trades(standard_events, transaction_cost_bps=transaction_cost_bps)
    confirmed_flow_trades, confirmed_flow_features = build_confirmed_flow_momentum_trades(
        standard_events,
        transaction_cost_bps=transaction_cost_bps,
        top_pct=0.20,
    )
    flow_trades, flow_features = build_flow_mean_reversion_trades(
        standard_events,
        transaction_cost_bps=transaction_cost_bps,
    )
    low_vol_deletion_trades, low_vol_deletion_features = build_low_vol_deletion_trades(
        standard_events,
        transaction_cost_bps=transaction_cost_bps,
        low_vol_rank_cutoff=0.50,
    )
    trades = pd.concat(
        [timing_trades, buy_hold_trades, confirmed_flow_trades, flow_trades, low_vol_deletion_trades],
        ignore_index=True,
        sort=False,
    )
    portfolio_daily, portfolio_summary = build_daily_portfolio_returns(trades)
    market_event_playbook = build_market_event_playbook(portfolio_summary)
    summary = summarize_trades(trades)
    coverage = coverage_summary(standard_events, trades)
    comparison = basic_vs_flow_comparison(summary)
    buy_hold_comparison = buy_hold_vs_best_comparison(summary)

    legs.to_csv(OUT_DIR / "raw_event_legs_all_countries.csv", index=False)
    events.to_csv(OUT_DIR / "cleaned_events_all_countries.csv", index=False)
    standard_events.to_csv(OUT_DIR / "standard_in_out_events_all_countries.csv", index=False)
    timing_trades.to_csv(OUT_DIR / "event_timing_trade_results.csv", index=False)
    buy_hold_trades.to_csv(OUT_DIR / "buy_hold_trades.csv", index=False)
    confirmed_flow_features.to_csv(OUT_DIR / "confirmed_flow_momentum_features.csv", index=False)
    confirmed_flow_trades.to_csv(OUT_DIR / "confirmed_flow_momentum_trades.csv", index=False)
    flow_features.to_csv(OUT_DIR / "flow_mean_reversion_features.csv", index=False)
    flow_trades.to_csv(OUT_DIR / "flow_mean_reversion_trades.csv", index=False)
    low_vol_deletion_features.to_csv(OUT_DIR / "developed_low_vol_deletion_features.csv", index=False)
    low_vol_deletion_trades.to_csv(OUT_DIR / "developed_low_vol_deletion_trades.csv", index=False)
    trades.to_csv(OUT_DIR / "event_window_trade_results.csv", index=False)
    portfolio_daily.to_csv(OUT_DIR / "portfolio_daily_returns.csv", index=False)
    portfolio_summary.to_csv(OUT_DIR / "portfolio_daily_summary.csv", index=False)
    market_event_playbook.to_csv(OUT_DIR / "market_event_rebalance_playbook.csv", index=False)
    summary.to_csv(OUT_DIR / "strategy_summary.csv", index=False)
    comparison.to_csv(OUT_DIR / "basic_vs_flow_strategy_comparison.csv", index=False)
    buy_hold_comparison.to_csv(OUT_DIR / "buy_hold_vs_best_strategy_comparison.csv", index=False)
    coverage.to_csv(OUT_DIR / "price_coverage_summary.csv", index=False)
    write_yahoo_ticker_diagnostics(standard_events)

    save_charts(summary, trades, portfolio_summary=portfolio_summary)
    save_market_event_playbook_chart(market_event_playbook)
    write_methodology(standard_events, coverage, summary)

    print(f"Saved outputs to {OUT_DIR}")
    print(f"Raw legs: {len(legs):,}")
    print(f"Cleaned events: {len(events):,}")
    print(f"Standard in/out events: {len(standard_events):,}")
    print(f"Priced event-timing trades: {len(timing_trades):,}")
    print(f"Buy/hold trades: {len(buy_hold_trades):,}")
    print(f"Confirmed-flow momentum trades: {len(confirmed_flow_trades):,}")
    print(f"Flow-neutral reversal trades: {len(flow_trades):,}")
    print(f"Developed low-vol deletion trades: {len(low_vol_deletion_trades):,}")
    print(f"Priced trades total: {len(trades):,}")
    if not summary.empty:
        print("Top strategy:")
        print(summary.head(1).to_string(index=False))
    if not portfolio_summary.empty:
        print("Top calendarized daily portfolio strategy:")
        print(portfolio_summary.head(1).to_string(index=False))
    print("Coverage:")
    print(coverage.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-country MSCI rebalance strategy pipeline outputs.")
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--parse-usa-from-pdfs", action="store_true", help="Do not reuse existing U.S. OCR legs.")
    args = parser.parse_args()
    run_pipeline(
        transaction_cost_bps=args.transaction_cost_bps,
        use_existing_usa=not args.parse_usa_from_pdfs,
    )


if __name__ == "__main__":
    main()
