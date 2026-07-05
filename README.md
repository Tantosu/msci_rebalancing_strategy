# MSCI Rebalance Strategy Project

This project preprocesses MSCI rebalance inclusion/deletion events for India,
Indonesia, Korea, and the United States, joins them to Yahoo Finance price and
volume data, and compares event-window trading rules against a simple buy/hold
benchmark.

## Market Universe

| Country | Stock market | Yahoo ticker format | Benchmark |
|---|---|---|---|
| India | NSE/BSE equities | `.NS` / `.BO` | `^NSEI` |
| Indonesia | Indonesia Stock Exchange | `.JK` | `^JKSE` |
| Korea | KOSPI/KOSDAQ | `.KS` / `.KQ` | `^KS11` |
| USA | NYSE/Nasdaq/NYSE American | no suffix | `SPY` |

## Main Question

For each market and event type, the project asks:

1. Is it better to simply buy/hold MSCI additions or short/hold MSCI deletions
   from announcement to effective date?
2. Or is there a better event timing rule around announcement/effective dates?

## Sharpe Delta

`best_minus_buy_hold_daily_sharpe` is:

```text
best active event-rule daily Sharpe - buy/hold daily Sharpe
```

It is calculated separately for each country and event type. Positive means the
active rebalance timing rule beat that market/event's own buy-hold benchmark on
calendarized daily Sharpe. Negative means buy-hold was better.

## Current Playbook

| Market | Event | Recommendation |
|---|---|---|
| India | Inclusion | `announce_d-5_to_d+0` |
| India | Deletion | `announce_d+0_to_d+5` |
| Indonesia | Inclusion | `buy_hold_announce_to_effective` |
| Indonesia | Deletion | `developed_low_vol_deletion_announce_d-1_to_d+1` |
| Korea | Inclusion | `buy_hold_announce_to_effective` |
| Korea | Deletion | `buy_hold_announce_to_effective` |
| USA | Inclusion | `announce_d+0_to_d+3` |
| USA | Deletion | `effective_d-3_to_d+0` |

The custom developed strategy filters deletion trades to the lower-volatility
half of the country/month deletion basket, then shorts from announcement day -1
to announcement day +1.

## Key Outputs

- `data/processed/msci_strategy/market_event_rebalance_playbook.csv`
- `data/processed/msci_strategy/portfolio_daily_summary.csv`
- `data/processed/msci_strategy/strategy_summary.csv`
- `data/processed/msci_strategy/price_coverage_summary.csv`
- `data/processed/msci_strategy/charts/market_event_buy_hold_vs_best_rule.png`

## Code

- `code/msci_multi_country_pipeline.py`: end-to-end preprocessing, strategy
  backtest, summaries, and charts.
- `code/fetch_yahoo_multicountry_data.py`: Yahoo ticker mapping and price/volume
  cache builder.

## Reproducing

Create a Python environment with:

```bash
pip install pandas numpy matplotlib pdfplumber yfinance
```

Then run:

```bash
python3 code/msci_multi_country_pipeline.py
```

If Yahoo price caches are missing, rebuild them first:

```bash
python3 code/fetch_yahoo_multicountry_data.py --countries India Indonesia Korea
python3 code/fetch_yahoo_multicountry_data.py --benchmarks-only --countries USA India Indonesia Korea
```

