import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from analyzer import ReviewAnalyzer

REVIEWS_PATH = "data/reviews/sample_reviews.csv"
PREDICTIONS_PATH = "data/reviews/predictions.csv"
SUMMARY_CSV = "reports/sentiment_summary.csv"
SUMMARY_PNG = "reports/sentiment_summary.png"

def load_reviews() -> pd.DataFrame:
    if not os.path.exists(REVIEWS_PATH):
        print(f"NAPAKA: {REVIEWS_PATH} ne obstaja")
        sys.exit(1)

    df = pd.read_csv(REVIEWS_PATH)
    print(f"Nalozenih {len(df)} reviews")
    return df


def predict_all(df: pd.DataFrame, analyzer: ReviewAnalyzer) -> pd.DataFrame:
    print(f"\nKlasificiram {len(df)} reviews...")

    texts = df["text"].astype(str).tolist()
    results = analyzer.predict_batch(texts)

    df["sentiment"] = [r["label"] for r in results]
    df["confidence"] = [r["score"] for r in results]
    df["score_negative"] = [r["scores"]["negative"] for r in results]
    df["score_neutral"] = [r["scores"]["neutral"] for r in results]
    df["score_positive"] = [r["scores"]["positive"] for r in results]

    return df


def aggregate_by_airline(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("airline").agg(
        n_reviews=("review_id", "count"),
        n_positive=("sentiment", lambda x: (x == "positive").sum()),
        n_neutral=("sentiment", lambda x: (x == "neutral").sum()),
        n_negative=("sentiment", lambda x: (x == "negative").sum()),
        avg_positive_score=("score_positive", "mean"),
        avg_negative_score=("score_negative", "mean"),
    ).reset_index()

    summary["sentiment_score"] = (
            summary["avg_positive_score"] - summary["avg_negative_score"]
    )

    # Pozitivni delež
    summary["pct_positive"] = (
            summary["n_positive"] / summary["n_reviews"] * 100
    ).round(1)
    summary["pct_negative"] = (
            summary["n_negative"] / summary["n_reviews"] * 100
    ).round(1)

    summary = summary.sort_values("sentiment_score", ascending=False)
    return summary


def plot_summary(summary: pd.DataFrame, output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Levi graf: stacked bar po airline
    ax1 = axes[0]
    airlines = summary["airline"].tolist()
    n_pos = summary["n_positive"].values
    n_neu = summary["n_neutral"].values
    n_neg = summary["n_negative"].values

    x = range(len(airlines))
    ax1.bar(x, n_pos, label="Positive", color="#28a745")
    ax1.bar(x, n_neu, bottom=n_pos, label="Neutral", color="#6c757d")
    ax1.bar(x, n_neg, bottom=n_pos + n_neu, label="Negative", color="#dc3545")
    ax1.set_xticks(x)
    ax1.set_xticklabels(airlines)
    ax1.set_ylabel("Število reviews")
    ax1.set_title("Sentiment distribucija po letalskih družbah")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Desni graf: sentiment score
    ax2 = axes[1]
    colors = ["#28a745" if s > 0 else "#dc3545" for s in summary["sentiment_score"]]
    ax2.barh(airlines[::-1], summary["sentiment_score"].values[::-1], color=colors[::-1])
    ax2.set_xlabel("Sentiment Score (-1 = negative, +1 = positive)")
    ax2.set_title("Povprečni sentiment score po družbah")
    ax2.axvline(x=0, color="black", linewidth=0.5)
    ax2.grid(axis="x", alpha=0.3)

    plt.suptitle("FlightInsight — Sentiment analiza letalskih reviews", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Graf shranjen: {output_path}")


def main():
    df = load_reviews()
    analyzer = ReviewAnalyzer()
    df = predict_all(df, analyzer)

    os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)
    df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"\nNapovedi shranjene: {PREDICTIONS_PATH}")

    print("\nDistribucija sentiment:")
    print(df["sentiment"].value_counts().to_string())

    # 5. Agregiraj po airline
    summary = aggregate_by_airline(df)

    os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nAgregat shranjen: {SUMMARY_CSV}")

    # 6. Vizualizacija
    plot_summary(summary, SUMMARY_PNG)

    # Top in bottom 3 družbe
    print("\nTop 3 družbe (najbolj pozitivne):")
    print(summary.head(3)[["airline", "n_reviews", "pct_positive", "sentiment_score"]].to_string(index=False))

    print("\nBottom 3 družbe (najbolj negativne):")
    print(summary.tail(3)[["airline", "n_reviews", "pct_negative", "sentiment_score"]].to_string(index=False))

    print("\nSentiment analiza končana!")

if __name__ == "__main__":
    main()