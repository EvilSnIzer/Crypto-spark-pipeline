# Databricks notebook source
# MAGIC %md
# MAGIC # Crypto Trades — PySpark Big-Data Pipeline (Databricks Community Edition)
# MAGIC
# MAGIC Re-architects a 211K+ trade record pipeline from Pandas to **PySpark on Databricks**.
# MAGIC
# MAGIC **Stages:** Bronze (typed Parquet) → Silver (window-function regimes) → Gold (Hive-partitioned aggregates) → Spark SQL analytics → Matplotlib viz.
# MAGIC
# MAGIC **How to run on Databricks Community Edition:**
# MAGIC 1. Create a free account at https://community.cloud.databricks.com
# MAGIC 2. Create a cluster (CE gives one free 15 GB cluster).
# MAGIC 3. Upload `data/raw/crypto_trades.csv` via *Data → DBFS → FileStore/crypto*.
# MAGIC 4. Import this `.py` file as a notebook (File → Import → File). It will appear with `# COMMAND ----------` cell breaks.
# MAGIC 5. Attach to the cluster and Run-All.

# COMMAND ----------
# MAGIC %md ## 0. Config & paths (DBFS on Databricks, local fallback otherwise)

# COMMAND ----------
import os

try:
    # Databricks-only
    dbutils.fs.ls("/FileStore")  # noqa: F821
    ON_DATABRICKS = True
except Exception:
    ON_DATABRICKS = False

if ON_DATABRICKS:
    RAW_CSV     = "dbfs:/FileStore/crypto/crypto_trades.csv"
    BRONZE_PATH = "dbfs:/FileStore/crypto/warehouse/bronze_trades"
    SILVER_PATH = "dbfs:/FileStore/crypto/warehouse/silver_trades"
    GOLD_ROOT   = "dbfs:/FileStore/crypto/warehouse/gold"
else:
    HERE = os.getcwd()
    ROOT = os.path.abspath(os.path.join(HERE, ".."))
    RAW_CSV     = f"file://{ROOT}/data/raw/crypto_trades.csv"
    BRONZE_PATH = f"file://{ROOT}/data/warehouse/bronze_trades"
    SILVER_PATH = f"file://{ROOT}/data/warehouse/silver_trades"
    GOLD_ROOT   = f"file://{ROOT}/data/warehouse/gold"

print("ON_DATABRICKS =", ON_DATABRICKS)
print("RAW_CSV       =", RAW_CSV)

# COMMAND ----------
# MAGIC %md ## 1. Bronze — typed ingest from CSV → partitioned Parquet

# COMMAND ----------
from pyspark.sql.types import (StructType, StructField, StringType,
                               DoubleType, TimestampType)

schema = StructType([
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

trades_raw = (spark.read.option("header", True).schema(schema).csv(RAW_CSV))
print(f"Ingested {trades_raw.count():,} rows; {len(trades_raw.columns)} cols")
trades_raw.printSchema()

(trades_raw.write
    .mode("overwrite")
    .partitionBy("trade_date")
    .parquet(BRONZE_PATH))

# COMMAND ----------
# MAGIC %md ## 2. Silver — window-function regimes (volatility + trend)

# COMMAND ----------
from pyspark.sql import Window
from pyspark.sql import functions as F

bronze = spark.read.parquet(BRONZE_PATH)

minute_bars = (bronze
    .withColumn("ts_min", F.date_trunc("minute", "ts"))
    .groupBy("symbol", "ts_min")
    .agg(F.avg("mid_price").alias("mid_min")))

w_sym = Window.partitionBy("symbol").orderBy("ts_min")
bars = (minute_bars
    .withColumn("prev_mid",   F.lag("mid_min", 1).over(w_sym))
    .withColumn("log_return", F.log(F.col("mid_min") / F.col("prev_mid")))
    .withColumn("sma_20",     F.avg("mid_min").over(w_sym.rowsBetween(-19, 0)))
    .withColumn("sma_60",     F.avg("mid_min").over(w_sym.rowsBetween(-59, 0)))
    .withColumn("vol_20",     F.stddev_pop("log_return").over(w_sym.rowsBetween(-19, 0))))

# Per-symbol vol terciles for the volatility regime label
sym_rows = [r["symbol"] for r in bars.select("symbol").distinct().collect()]
thr_rows = []
for s in sym_rows:
    q = (bars.filter((F.col("symbol") == s) & F.col("vol_20").isNotNull())
              .approxQuantile("vol_20", [0.33, 0.66], 0.01))
    thr_rows.append((s, q[0] if q else 0.0, q[1] if len(q) > 1 else 0.0))
thr_df = spark.createDataFrame(thr_rows, ["symbol", "vol_lo", "vol_hi"])

bars_lbl = (bars.join(F.broadcast(thr_df), on="symbol", how="left")
    .withColumn("vol_regime",
        F.when(F.col("vol_20").isNull(), F.lit("unknown"))
         .when(F.col("vol_20") <= F.col("vol_lo"), F.lit("low"))
         .when(F.col("vol_20") <= F.col("vol_hi"), F.lit("medium"))
         .otherwise(F.lit("high")))
    .withColumn("trend_regime",
        F.when(F.col("sma_20").isNull() | F.col("sma_60").isNull(), F.lit("unknown"))
         .when(F.col("sma_20") > F.col("sma_60") * 1.001, F.lit("bull"))
         .when(F.col("sma_20") < F.col("sma_60") * 0.999, F.lit("bear"))
         .otherwise(F.lit("chop")))
    .drop("vol_lo", "vol_hi"))

silver = (bronze
    .withColumn("ts_min", F.date_trunc("minute", "ts"))
    .join(bars_lbl.select("symbol", "ts_min", "vol_20", "sma_20",
                          "sma_60", "vol_regime", "trend_regime"),
          on=["symbol", "ts_min"], how="left")
    .withColumn("net_pnl", F.col("realized_pnl")))

(silver.write.mode("overwrite")
    .partitionBy("trade_date", "symbol")
    .parquet(SILVER_PATH))

silver_read = spark.read.parquet(SILVER_PATH)
silver_read.createOrReplaceTempView("trades")
display(silver_read.limit(10))  # noqa: F821 (display = Databricks builtin)

# COMMAND ----------
# MAGIC %md ## 3. Spark SQL — Hive-style regime aggregations + window ranking

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT tier, vol_regime,
# MAGIC        COUNT(*) AS n_trades,
# MAGIC        ROUND(SUM(net_pnl), 2)   AS total_pnl,
# MAGIC        ROUND(AVG(net_pnl), 4)   AS avg_pnl,
# MAGIC        ROUND(SUM(fee_amount),2) AS fees_paid
# MAGIC FROM trades
# MAGIC WHERE vol_regime <> 'unknown'
# MAGIC GROUP BY tier, vol_regime
# MAGIC ORDER BY tier, vol_regime

# COMMAND ----------
# MAGIC %sql
# MAGIC -- DENSE_RANK window function: top accounts per (vol_regime, symbol)
# MAGIC WITH acct AS (
# MAGIC   SELECT account_id, tier, vol_regime, symbol,
# MAGIC          SUM(net_pnl)  AS total_pnl,
# MAGIC          COUNT(*)      AS n_trades
# MAGIC   FROM trades
# MAGIC   WHERE vol_regime <> 'unknown'
# MAGIC   GROUP BY account_id, tier, vol_regime, symbol
# MAGIC )
# MAGIC SELECT * FROM (
# MAGIC   SELECT account_id, tier, vol_regime, symbol, total_pnl, n_trades,
# MAGIC          DENSE_RANK() OVER (PARTITION BY vol_regime, symbol
# MAGIC                             ORDER BY total_pnl DESC) AS pnl_rank
# MAGIC   FROM acct
# MAGIC ) WHERE pnl_rank <= 3
# MAGIC ORDER BY vol_regime, symbol, pnl_rank

# COMMAND ----------
# MAGIC %md ## 4. Gold — write Hive-partitioned aggregates to Parquet

# COMMAND ----------
agg_vol = spark.sql("""
    SELECT vol_regime, symbol,
           COUNT(*)            AS n_trades,
           SUM(notional)       AS gross_notional,
           SUM(fee_amount)     AS total_fees,
           SUM(net_pnl)        AS total_pnl,
           AVG(net_pnl)        AS avg_pnl_per_trade,
           STDDEV_POP(net_pnl) AS pnl_stddev
    FROM trades WHERE vol_regime <> 'unknown'
    GROUP BY vol_regime, symbol
""")
(agg_vol.write.mode("overwrite").partitionBy("vol_regime")
    .parquet(f"{GOLD_ROOT}/agg_by_vol_regime"))

ranked = spark.sql("""
    WITH acct AS (
      SELECT account_id, tier, vol_regime, symbol,
             SUM(net_pnl) AS total_pnl, COUNT(*) AS n_trades,
             SUM(notional) AS gross_notional, SUM(fee_amount) AS total_fees
      FROM trades WHERE vol_regime <> 'unknown'
      GROUP BY account_id, tier, vol_regime, symbol
    )
    SELECT *,
      DENSE_RANK() OVER (PARTITION BY vol_regime, symbol ORDER BY total_pnl DESC) AS pnl_rank,
      ROW_NUMBER() OVER (PARTITION BY vol_regime, symbol ORDER BY total_pnl DESC) AS pnl_row_num,
      SUM(total_pnl) OVER (PARTITION BY vol_regime, symbol ORDER BY total_pnl DESC
                           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_pnl
    FROM acct
""")
(ranked.write.mode("overwrite").partitionBy("vol_regime", "symbol")
    .parquet(f"{GOLD_ROOT}/account_ranking_by_regime"))

# COMMAND ----------
# MAGIC %md ## 5. Visualize via display() and matplotlib

# COMMAND ----------
display(agg_vol.orderBy("vol_regime", "symbol"))  # noqa: F821

# COMMAND ----------
import matplotlib.pyplot as plt

pdf = (agg_vol.toPandas()
       .pivot(index="symbol", columns="vol_regime", values="total_pnl")
       .fillna(0)[["low", "medium", "high"]])
ax = pdf.plot(kind="bar", figsize=(10, 5),
              color=["#3a86ff", "#ffb703", "#e63946"], edgecolor="black")
ax.set_title("Net PnL by Symbol × Volatility Regime"); ax.axhline(0, color="black", lw=0.6)
plt.tight_layout(); plt.show()

# COMMAND ----------
# MAGIC %md ## 6. Verify Hive-style partition layout

# COMMAND ----------
if ON_DATABRICKS:
    display(dbutils.fs.ls(f"{GOLD_ROOT}/account_ranking_by_regime/vol_regime=high"))  # noqa: F821
else:
    import subprocess
    print(subprocess.check_output(
        ["find", GOLD_ROOT.replace("file://", ""), "-maxdepth", "3", "-type", "d"]
    ).decode())
