"""
End-to-end PySpark big-data pipeline for the 211K crypto trades dataset.

Stages
------
1. Ingest CSV with explicit schema (no inferSchema in prod -> stable, fast).
2. Bronze: write raw-as-typed to Parquet, partitioned by trade_date.
3. Silver: enrichment + 3 Spark SQL window-function analytics
     a) Volatility regimes  (rolling stddev of returns per symbol -> quantile bucket)
     b) Trend regimes       (SMA20/SMA60 crossover + slope)
     c) Per-account ranking (dense_rank over PnL per regime per symbol)
4. Gold: regime-based aggregation tables written to partitioned Parquet
         (Hive-style: /regime=high/symbol=BTC-USD/part-*.parquet).
5. Spark SQL queries against the registered temp views.

Designed so the same script runs locally (master=local[*]) and on
Databricks Community Edition without modification (paths are parametric).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, TimestampType,
)

# ----------------------------------------------------------------------------
# 0. Config
# ----------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.abspath(os.path.join(HERE, ".."))

TRADE_SCHEMA = StructType([
    StructField("trade_id",     StringType(),    False),
    StructField("ts",           TimestampType(), False),
    StructField("account_id",   StringType(),    False),
    StructField("tier",         StringType(),    False),
    StructField("symbol",       StringType(),    False),
    StructField("venue",        StringType(),    False),
    StructField("side",         StringType(),    False),
    StructField("quantity",     DoubleType(),    False),
    StructField("fill_price",   DoubleType(),    False),
    StructField("mid_price",    DoubleType(),    False),
    StructField("notional",     DoubleType(),    False),
    StructField("fee_bps",      DoubleType(),    False),
    StructField("fee_amount",   DoubleType(),    False),
    StructField("realized_pnl", DoubleType(),    False),
    StructField("trade_date",   StringType(),    False),
])


# ----------------------------------------------------------------------------
# 1. Spark session
# ----------------------------------------------------------------------------

def build_spark(app_name: str = "crypto-spark-pipeline") -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.compression.codec", "snappy")
        # Adaptive Query Execution -- realistic prod config
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ----------------------------------------------------------------------------
# 2. Bronze: typed ingest -> partitioned Parquet
# ----------------------------------------------------------------------------

def ingest_bronze(spark: SparkSession, csv_path: str, bronze_path: str) -> DataFrame:
    print(f"\n[BRONZE] Reading {csv_path}")
    df = (
        spark.read
        .option("header", True)
        .schema(TRADE_SCHEMA)
        .csv(csv_path)
    )
    n = df.count()
    print(f"[BRONZE] Ingested rows: {n:,}")
    print(f"[BRONZE] Writing -> {bronze_path}  (partitioned by trade_date)")
    (
        df.write
        .mode("overwrite")
        .partitionBy("trade_date")
        .parquet(bronze_path)
    )
    return df


# ----------------------------------------------------------------------------
# 3. Silver: enrich + window-function regimes
# ----------------------------------------------------------------------------

def build_silver(spark: SparkSession, bronze_path: str, silver_path: str) -> DataFrame:
    """Add returns, rolling vol, SMA, regime labels via window functions."""
    print(f"\n[SILVER] Reading bronze parquet")
    bronze = spark.read.parquet(bronze_path)

    # Resample-ish: compute symbol-minute mid prices (one row per symbol per minute)
    # then attach back. This is the canonical pattern for time-series in Spark.
    minute_bars = (
        bronze
        .withColumn("ts_min", F.date_trunc("minute", F.col("ts")))
        .groupBy("symbol", "ts_min")
        .agg(F.avg("mid_price").alias("mid_min"))
    )

    w_sym = Window.partitionBy("symbol").orderBy("ts_min")
    w_sym_20 = w_sym.rowsBetween(-19, 0)
    w_sym_60 = w_sym.rowsBetween(-59, 0)

    bars_enriched = (
        minute_bars
        .withColumn("prev_mid",     F.lag("mid_min", 1).over(w_sym))
        .withColumn("log_return",   F.log(F.col("mid_min") / F.col("prev_mid")))
        .withColumn("sma_20",       F.avg("mid_min").over(w_sym_20))
        .withColumn("sma_60",       F.avg("mid_min").over(w_sym_60))
        .withColumn("vol_20",       F.stddev_pop("log_return").over(w_sym_20))
    )

    # ----- Volatility regime: per-symbol quantile bucket of vol_20 -----
    # approxQuantile is the canonical scalable way in Spark.
    vol_thresholds: dict[str, tuple[float, float]] = {}
    symbols = [r["symbol"] for r in bars_enriched.select("symbol").distinct().collect()]
    for sym in symbols:
        q = (
            bars_enriched
            .filter(F.col("symbol") == sym)
            .filter(F.col("vol_20").isNotNull())
            .approxQuantile("vol_20", [0.33, 0.66], 0.01)
        )
        vol_thresholds[sym] = (q[0], q[1]) if len(q) == 2 else (0.0, 0.0)

    # Broadcast the thresholds as a tiny DF
    thr_rows = [(s, lo, hi) for s, (lo, hi) in vol_thresholds.items()]
    thr_df = spark.createDataFrame(thr_rows, ["symbol", "vol_lo", "vol_hi"])

    bars_with_regime = (
        bars_enriched.join(F.broadcast(thr_df), on="symbol", how="left")
        .withColumn(
            "vol_regime",
            F.when(F.col("vol_20").isNull(), F.lit("unknown"))
             .when(F.col("vol_20") <= F.col("vol_lo"), F.lit("low"))
             .when(F.col("vol_20") <= F.col("vol_hi"), F.lit("medium"))
             .otherwise(F.lit("high")),
        )
        .withColumn(
            "trend_regime",
            F.when(F.col("sma_20").isNull() | F.col("sma_60").isNull(), F.lit("unknown"))
             .when(F.col("sma_20") > F.col("sma_60") * 1.001, F.lit("bull"))
             .when(F.col("sma_20") < F.col("sma_60") * 0.999, F.lit("bear"))
             .otherwise(F.lit("chop")),
        )
        .drop("vol_lo", "vol_hi")
    )

    # Join regime labels back to every trade
    enriched = (
        bronze
        .withColumn("ts_min", F.date_trunc("minute", F.col("ts")))
        .join(
            bars_with_regime.select("symbol", "ts_min", "vol_20", "sma_20",
                                    "sma_60", "vol_regime", "trend_regime"),
            on=["symbol", "ts_min"],
            how="left",
        )
        .withColumn("net_pnl", F.col("realized_pnl"))  # already net of fee
    )

    print(f"[SILVER] Writing -> {silver_path}  (partitioned by trade_date, symbol)")
    (
        enriched.write
        .mode("overwrite")
        .partitionBy("trade_date", "symbol")
        .parquet(silver_path)
    )
    return enriched


# ----------------------------------------------------------------------------
# 4. Gold: regime-based aggregations + per-account ranking
# ----------------------------------------------------------------------------

def build_gold(spark: SparkSession, silver_path: str, gold_root: str) -> None:
    print(f"\n[GOLD] Reading silver parquet")
    silver = spark.read.parquet(silver_path)
    silver.createOrReplaceTempView("trades")

    # --- (a) per-regime per-symbol aggregation --------------------------------
    agg_vol = spark.sql("""
        SELECT
            vol_regime,
            symbol,
            COUNT(*)                AS n_trades,
            SUM(notional)           AS gross_notional,
            SUM(fee_amount)         AS total_fees,
            SUM(net_pnl)            AS total_pnl,
            AVG(net_pnl)            AS avg_pnl_per_trade,
            STDDEV_POP(net_pnl)     AS pnl_stddev
        FROM trades
        WHERE vol_regime <> 'unknown'
        GROUP BY vol_regime, symbol
        ORDER BY vol_regime, symbol
    """)
    out = os.path.join(gold_root, "agg_by_vol_regime")
    print(f"[GOLD] Writing -> {out}  (partitioned by vol_regime)")
    agg_vol.write.mode("overwrite").partitionBy("vol_regime").parquet(out)

    # --- (b) per-account ranking via window function ---------------------------
    # dense_rank top accounts by net_pnl within each (vol_regime, symbol)
    acct_perf = spark.sql("""
        SELECT
            account_id,
            tier,
            vol_regime,
            symbol,
            COUNT(*)        AS n_trades,
            SUM(net_pnl)    AS total_pnl,
            SUM(notional)   AS gross_notional,
            SUM(fee_amount) AS total_fees
        FROM trades
        WHERE vol_regime <> 'unknown'
        GROUP BY account_id, tier, vol_regime, symbol
    """)
    acct_perf.createOrReplaceTempView("acct_perf")

    ranked = spark.sql("""
        SELECT
            account_id, tier, vol_regime, symbol,
            n_trades, total_pnl, gross_notional, total_fees,
            DENSE_RANK() OVER (
                PARTITION BY vol_regime, symbol
                ORDER BY total_pnl DESC
            ) AS pnl_rank,
            ROW_NUMBER() OVER (
                PARTITION BY vol_regime, symbol
                ORDER BY total_pnl DESC
            ) AS pnl_row_num,
            SUM(total_pnl) OVER (
                PARTITION BY vol_regime, symbol
                ORDER BY total_pnl DESC
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cumulative_pnl
        FROM acct_perf
    """)
    out = os.path.join(gold_root, "account_ranking_by_regime")
    print(f"[GOLD] Writing -> {out}  (partitioned by vol_regime, symbol)")
    ranked.write.mode("overwrite").partitionBy("vol_regime", "symbol").parquet(out)

    # --- (c) daily PnL by tier + trend regime ---------------------------------
    daily = spark.sql("""
        SELECT
            trade_date,
            tier,
            trend_regime,
            COUNT(*)      AS n_trades,
            SUM(net_pnl)  AS total_pnl,
            SUM(notional) AS gross_notional
        FROM trades
        WHERE trend_regime <> 'unknown'
        GROUP BY trade_date, tier, trend_regime
        ORDER BY trade_date, tier, trend_regime
    """)
    out = os.path.join(gold_root, "daily_pnl_by_tier_trend")
    print(f"[GOLD] Writing -> {out}  (partitioned by trade_date)")
    daily.write.mode("overwrite").partitionBy("trade_date").parquet(out)

    # --- Sanity prints --------------------------------------------------------
    print("\n[GOLD] Sample: aggregation by volatility regime")
    agg_vol.show(10, truncate=False)
    print("\n[GOLD] Sample: top 3 accounts per (vol_regime, symbol)")
    spark.sql("""
        SELECT * FROM (
          SELECT * FROM (
            SELECT account_id, tier, vol_regime, symbol, total_pnl, pnl_rank
            FROM (
                SELECT account_id, tier, vol_regime, symbol, total_pnl,
                       DENSE_RANK() OVER (PARTITION BY vol_regime, symbol
                                          ORDER BY total_pnl DESC) AS pnl_rank
                FROM acct_perf
            ) WHERE pnl_rank <= 3
          )
        )
        ORDER BY vol_regime, symbol, pnl_rank
        LIMIT 20
    """).show(20, truncate=False)


# ----------------------------------------------------------------------------
# 5. Driver
# ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=DEFAULT_ROOT,
                   help="Project root containing data/ folder.")
    args = p.parse_args(argv)

    csv_path    = os.path.join(args.root, "data", "raw", "crypto_trades.csv")
    bronze_path = os.path.join(args.root, "data", "warehouse", "bronze_trades")
    silver_path = os.path.join(args.root, "data", "warehouse", "silver_trades")
    gold_root   = os.path.join(args.root, "data", "warehouse", "gold")

    spark = build_spark()
    t0 = time.time()
    ingest_bronze(spark, csv_path, bronze_path)
    build_silver(spark, bronze_path, silver_path)
    build_gold(spark, silver_path, gold_root)
    elapsed = time.time() - t0
    print(f"\n[PIPELINE] Done in {elapsed:.1f}s")
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
