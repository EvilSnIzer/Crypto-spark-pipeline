# Interview-defense notes

Short, honest answers to questions an interviewer is most likely to ask after
seeing the resume bullets. Read these once before any conversation.

---

## "Walk me through your PySpark pipeline."

Three medallion layers — Bronze, Silver, Gold — written to partitioned Parquet.
Bronze is a typed ingest of the raw 211K-row CSV, partitioned by `trade_date`.
Silver enriches each trade with a volatility regime and a trend regime,
computed via Spark window functions over per-symbol minute bars: I compute
`vol_20` as a 20-row `stddev_pop(log_return)` over a `Window.partitionBy(symbol).orderBy(ts_min).rowsBetween(-19, 0)`, then bucket per-symbol terciles via `approxQuantile`.
Gold is the analyst layer: per-(vol_regime, symbol) aggregations and a
`DENSE_RANK() OVER (PARTITION BY vol_regime, symbol ORDER BY total_pnl DESC)`
account leaderboard, written out partitioned by `vol_regime` and `symbol` — that
gives Hive-style `vol_regime=high/symbol=BTC-USD/` directories that Spark can
prune at read time.

## "Why Parquet, why partition by those keys?"

Parquet is columnar with snappy compression — predicate and column pushdown
means `SELECT total_pnl FROM acct_rank WHERE vol_regime='high'` only touches one
partition directory and reads one column, not the whole dataset. I chose
`vol_regime` and `symbol` as partition keys because those are the actual filter
predicates analysts use. I deliberately did **not** partition by
`account_id` — 32 values would create 32 tiny partitions; partition cardinality
should roughly match expected query selectivity without exploding small-file
counts.

## "Spark vs. Hive — what's the relationship?"

Spark SQL implements the **HiveQL dialect** and reads Hive's metastore and
table formats. My `partitionBy("vol_regime", "symbol")` produces the exact
`/key=value/` directory layout Hive uses, so I could `CREATE EXTERNAL TABLE …
PARTITIONED BY (vol_regime string, symbol string)` over this Parquet and Hive
would read it without any rewrite. That's why the resume says "Spark SQL which
is syntactically close to Hive" — `WINDOW`, `LATERAL VIEW`, `GROUPING SETS`,
etc. all work the same.

## "What's a window function and where did you use one?"

A window function computes per-row aggregates over a defined frame of related
rows without collapsing the rows like `GROUP BY` would. I used:

- `lag(mid_min, 1) OVER (PARTITION BY symbol ORDER BY ts_min)` — to compute log returns.
- `stddev_pop(log_return) OVER (… ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)` — rolling 20-bar volatility for the regime label.
- `avg(mid_min) OVER (… ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)` — SMA60 for the trend label.
- `DENSE_RANK() OVER (PARTITION BY vol_regime, symbol ORDER BY total_pnl DESC)` — top-account leaderboards.
- `SUM(total_pnl) OVER (… ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)` — cumulative PnL.

## "Did this actually need Spark?"

Honest answer: at 211K rows × 15 cols × 28 MB, no — a laptop running pandas is
fine. The point of this build is to prove I can **design** the pipeline so
that scaling to 211 million or 2.1 billion rows is a configuration change
(more executors, more shuffle partitions, broadcast thresholds) and not a
rewrite. The same script runs on `local[*]` and on a Databricks cluster
unchanged because paths and master are parametric.

## "What would you change for a real prod deployment?"

- Move the medallion tables into **Delta Lake** so we get ACID, time travel, and `MERGE INTO` for late-arriving trades.
- Replace the per-symbol Python loop that computes `approxQuantile` thresholds with a single `percentile_approx` window expression so it's one Spark job, not N.
- Wire up a job orchestrator — Airflow, Databricks Workflows, or `dbt-spark` — instead of running `python pipeline.py` directly.
- Schema enforcement / DLT expectations on the bronze-to-silver hop.
- Move secrets to a vault and stop hard-coding paths.
- Add unit tests for the regime-classification UDFs using `chispa` or local SparkSession fixtures.

## "Where might this break?"

- `approxQuantile` is sample-based — with a tiny per-symbol filter you can get bad thresholds. I'd switch to `percentile_approx` with a higher accuracy setting once data is bigger.
- My PnL definition (forward 30-min mid vs fill) is a proxy, not a true mark-to-market. Fine for a demo, would need a proper market-data join in prod.
- I'm doing a `groupBy("symbol", "ts_min")` then joining back to trades — that's a self-join shuffle. At true big-data scale I'd push the regime label down into a streaming feature store keyed by `(symbol, minute)` instead.
