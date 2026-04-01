from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "modeling" / "outputs"

pred_path = OUTPUT_DIR / "xgb_test_predictions_no_clip.csv"
imp_path = OUTPUT_DIR / "xgb_feature_importance_no_clip.csv"

pred_df = pd.read_csv(pred_path)
imp_df = pd.read_csv(imp_path)

plt.figure(figsize=(8, 5))

x = range(len(pred_df))
width = 0.35

bars_true = plt.bar(
    [i - width / 2 for i in x],
    pred_df["y_true"],
    width=width,
    label="True"
)
bars_pred = plt.bar(
    [i + width / 2 for i in x],
    pred_df["y_pred"],
    width=width,
    label="Predicted"
)

plt.xticks(list(x), pred_df["year"].astype(str))
plt.xlabel("Year")
plt.ylabel("Life Expectancy")
plt.title("Random Forest: True vs Predicted Life Expectancy (Test Set)")
plt.legend()

y_min = min(pred_df["y_true"].min(), pred_df["y_pred"].min()) - 0.2
y_max = max(pred_df["y_true"].max(), pred_df["y_pred"].max()) + 0.2
plt.ylim(y_min, y_max)

for bar in bars_true:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.01,
        f"{h:.2f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

for bar in bars_pred:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.01,
        f"{h:.2f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "xgb_true_vs_pred_bar_no_clip.png", dpi=300)
plt.show()

top10 = imp_df.head(10).copy().sort_values(by="importance", ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(top10["feature"], top10["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest: Top 10 Feature Importances (no_clip)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "xgb_feature_importance_no_clip.png", dpi=300)
plt.show()