"""
Stand-alone Spark SQL query runner against the gold/silver Parquet warehouse.

Demonstrates:
  - Reading partitioned Parquet via Spark SQL (predicate pushdown / partition pruning)
  - Hive-style /key=value/ partition discovery
  - Window functions (RANK, LAG, AVG OVER) at the SQL layer
  - EXPLAIN plans showing partition filter usage
"""
from __future__ import annotations
import os
from pyspark.sql import SparkSession

HERE = os.path.dirname(os.path.abspath(__file__))
WAREHOUSE = os.path.abspath(os.path.join(HERE, "..", "data", "warehouse"))


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("crypto-spark-sql-queries")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    spark.read.parquet(os.path.join(WAREHOUSE, "silver_trades")) \
         .createOrReplaceTempView("trades")
    spark.read.parquet(os.path.join(WAREHOUSE, "gold", "agg_by_vol_regime")) \
         .createOrReplaceTempView("agg_vol")
    spark.read.parquet(os.path.join(WAREHOUSE, "gold", "account_ranking_by_regime")) \
         .createOrReplaceTempView("acct_rank")
    spark.read.parquet(os.path.join(WAREHOUSE, "gold", "daily_pnl_by_tier_trend")) \
         .createOrReplaceTempView("daily_tier")

    queries: list[tuple[str, str]] = [
        ("Q1: PnL by tier across volatility regimes", """
            SELECT tier, vol_regime,
                   COUNT(*)       AS n_trades,
                   ROUND(SUM(net_pnl), 2)   AS total_pnl,
                   ROUND(AVG(net_pnl), 4)   AS avg_pnl,
                   ROUND(SUM(fee_amount),2) AS fees_paid
            FROM trades
            WHERE vol_regime <> 'unknown'
            GROUP BY tier, vol_regime
            ORDER BY tier, vol_regime
        """),
        ("Q2: Top-5 accounts by PnL in HIGH vol (partition pruned)", """
            SELECT account_id, tier, symbol,
                   ROUND(total_pnl, 2) AS total_pnl,
                   pnl_rank
            FROM acct_rank
            WHERE vol_regime = 'high' AND pnl_rank <= 5
            ORDER BY symbol, pnl_rank
        """),
        ("Q3: Symbol-day momentum using LAG (window function over SQL)", """
            SELECT trade_date, symbol,
                   ROUND(SUM(net_pnl), 2) AS pnl,
                   ROUND(LAG(SUM(net_pnl), 1) OVER
                         (PARTITION BY symbol ORDER BY trade_date), 2) AS prev_day_pnl,
                   ROUND(SUM(net_pnl) - LAG(SUM(net_pnl), 1) OVER
                         (PARTITION BY symbol ORDER BY trade_date), 2) AS day_change
            FROM trades
            WHERE symbol IN ('BTC-USD','ETH-USD')
              AND trade_date BETWEEN '2024-01-15' AND '2024-01-20'
            GROUP BY trade_date, symbol
            ORDER BY symbol, trade_date
        """),
        ("Q4: Per-symbol 7-day moving PnL (AVG OVER ROWS BETWEEN)", """
            WITH daily AS (
              SELECT trade_date, symbol, SUM(net_pnl) AS pnl
              FROM trades GROUP BY trade_date, symbol
            )
            SELECT trade_date, symbol,
                   ROUND(pnl, 2)                                            AS daily_pnl,
                   ROUND(AVG(pnl) OVER (PARTITION BY symbol ORDER BY trade_date
                                        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2)
                                                                            AS ma7_pnl
            FROM daily
            WHERE symbol = 'BTC-USD'
            ORDER BY trade_date
            LIMIT 15
        """),
        ("Q5: Tier dominance in HIGH vol regime", """
            SELECT tier,
                   COUNT(DISTINCT account_id)         AS n_accounts,
                   SUM(n_trades)                      AS n_trades,
                   ROUND(SUM(total_pnl), 2)           AS total_pnl,
                   ROUND(SUM(total_pnl)/SUM(n_trades),4) AS pnl_per_trade
            FROM acct_rank
            WHERE vol_regime = 'high'
            GROUP BY tier
            ORDER BY total_pnl DESC
        """),
    ]

    for title, sql in queries:
        print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)
        df = spark.sql(sql)
        df.show(50, truncate=False)

    # Show partition pruning in action
    print("\n" + "=" * 80 + "\nEXPLAIN: partition pruning on acct_rank (vol_regime='high')\n" + "=" * 80)
    spark.sql("SELECT * FROM acct_rank WHERE vol_regime='high'").explain(True)

    spark.stop()


if __name__ == "__main__":
    main()
