"""
Read the Gold Parquet tables back with PySpark, collect small aggregates,
and render matplotlib charts for the README / interview deck.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

HERE = os.path.dirname(os.path.abspath(__file__))
WAREHOUSE = os.path.abspath(os.path.join(HERE, "..", "data", "warehouse"))
CHART_DIR = os.path.abspath(os.path.join(HERE, "..", "output", "charts"))
os.makedirs(CHART_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


def main() -> None:
    spark = (SparkSession.builder.appName("crypto-charts")
             .master("local[*]").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    silver = spark.read.parquet(os.path.join(WAREHOUSE, "silver_trades"))
    agg_vol = spark.read.parquet(os.path.join(WAREHOUSE, "gold", "agg_by_vol_regime"))
    daily = spark.read.parquet(os.path.join(WAREHOUSE, "gold", "daily_pnl_by_tier_trend"))
    rank = spark.read.parquet(os.path.join(WAREHOUSE, "gold", "account_ranking_by_regime"))

    # ---- Chart 1: total PnL by symbol x vol_regime -------------------------
    pdf = (agg_vol
           .groupBy("symbol", "vol_regime").agg(F.sum("total_pnl").alias("pnl"))
           .toPandas()
           .pivot(index="symbol", columns="vol_regime", values="pnl")
           .fillna(0.0)[["low", "medium", "high"]])
    ax = pdf.plot(kind="bar", figsize=(10, 5),
                  color=["#3a86ff", "#ffb703", "#e63946"], edgecolor="black")
    ax.set_title("Total Net PnL by Symbol × Volatility Regime")
    ax.set_ylabel("Net PnL (USD)")
    ax.axhline(0, color="black", linewidth=0.6)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    p = os.path.join(CHART_DIR, "01_pnl_by_symbol_vol_regime.png")
    plt.savefig(p, dpi=140); plt.close()
    print("wrote", p)

    # ---- Chart 2: daily PnL by tier ----------------------------------------
    pdf = (daily.groupBy("trade_date", "tier").agg(F.sum("total_pnl").alias("pnl"))
                .toPandas())
    pdf["trade_date"] = pdf["trade_date"].astype("datetime64[ns]")
    pdf = pdf.pivot(index="trade_date", columns="tier", values="pnl").fillna(0).sort_index()
    pdf_cum = pdf.cumsum()
    fig, ax = plt.subplots(figsize=(11, 5))
    for col, color in zip(pdf_cum.columns,
                          ["#06d6a0", "#118ab2", "#ef476f"]):
        ax.plot(pdf_cum.index, pdf_cum[col], label=col, linewidth=2, color=color)
    ax.set_title("Cumulative Net PnL by Account Tier (60-day window)")
    ax.set_ylabel("Cumulative PnL (USD)")
    ax.legend()
    plt.tight_layout()
    p = os.path.join(CHART_DIR, "02_cumulative_pnl_by_tier.png")
    plt.savefig(p, dpi=140); plt.close()
    print("wrote", p)

    # ---- Chart 3: trade-count share per regime -----------------------------
    pdf = (silver.filter(F.col("vol_regime") != "unknown")
                  .groupBy("vol_regime").count().toPandas()
                  .sort_values("vol_regime"))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(pdf["count"], labels=pdf["vol_regime"],
           colors=["#e63946", "#3a86ff", "#ffb703"],
           autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor": "white"})
    ax.set_title("Trade-Count Share by Volatility Regime")
    plt.tight_layout()
    p = os.path.join(CHART_DIR, "03_trade_share_by_regime.png")
    plt.savefig(p, dpi=140); plt.close()
    print("wrote", p)

    # ---- Chart 4: top accounts in HIGH vol BTC -----------------------------
    pdf = (rank.filter((F.col("vol_regime") == "high") & (F.col("symbol") == "BTC-USD"))
                .orderBy("pnl_rank").limit(10).toPandas())
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#06d6a0" if t == "market_maker" else
              "#118ab2" if t == "prop" else "#ef476f" for t in pdf["tier"]]
    ax.barh(pdf["account_id"], pdf["total_pnl"], color=colors, edgecolor="black")
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_title("Top 10 Accounts by Net PnL — BTC-USD in HIGH vol regime")
    ax.set_xlabel("Net PnL (USD)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in ["#06d6a0", "#118ab2", "#ef476f"]]
    ax.legend(handles, ["market_maker", "prop", "retail"], loc="lower right")
    plt.tight_layout()
    p = os.path.join(CHART_DIR, "04_top_accounts_btc_high_vol.png")
    plt.savefig(p, dpi=140); plt.close()
    print("wrote", p)

    spark.stop()


if __name__ == "__main__":
    main()
