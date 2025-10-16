# Vehicles Price Modeling – README

A compact, end‑to‑end workflow to clean a used‑vehicles dataset, explore it quickly, build a linear regression model, and export visuals (EDA plots, coefficient bars)

## Project Structure

```
vehicles-price-model/
├─ data/
│  ├─ raw/vehicles.csv
│  └─ processed/vehicles_clean.parquet
├─ notebooks/
│  └─ 00_quickstart.ipynb
├─ src/
│  ├─ load_and_filter.py
│  ├─ eda.py
│  ├─ prep.py
│  ├─ model.py
│  ├─ visuals.py
│  └─ report.py
├─ outputs/
│  ├─ eda/
│  │  ├─ hist_price.png
│  │  ├─ hist_year.png
│  │  ├─ hist_odometer.png
│  │  ├─ price_vs_year.png
│  │  └─ price_vs_odometer.png
│  ├─ model/
│  │  ├─ metrics.json
│  │  └─ coefficients.csv
│  ├─ charts/
│  │  └─ coef_top_pos_neg.png
│  ├─ dashboard/
│  │  └─ dashboard.html
│  └─ deck/
│     └─ Vehicles_Model_Summary.pptx
├─ README.md
└─ requirements.txt
```


## 1) Load & Sanity Filter

**Input:** `data/raw/vehicles.csv` (must include at least: `price`, `year`, `odometer` plus the categorical columns listed in §3)

Rules:
- Keep realistic prices: **$101–$99,999**
- Cap mileage: **odometer < 500,000**
- Drop duplicates (row‑wise or by domain keys like `['year','manufacturer','model','odometer','price','state']`)

**Example (Python):**
```python
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path("data/raw/vehicles.csv"))
df = df[(df["price"].between(101, 99999, inclusive="both")) & (df["odometer"] < 500_000)]
# Optional: stricter year sanity (e.g., 1980–current year)
df = df[(df["year"] >= 1980) & (df["year"] <= pd.Timestamp.now().year + 1)]
df = df.drop_duplicates()
df.to_parquet("data/processed/vehicles_clean.parquet", index=False)
print(df.shape)
```

---

## 2) Quick EDA

Make basic distributions and simple relationships:

- Histograms: **price**, **year**, **odometer**
- Scatter + trendline: **Price vs Year**
- Scatter + trendline: **Price vs Odometer**

**Example (matplotlib):**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet("data/processed/vehicles_clean.parquet")

for col, fn in [("price","hist_price.png"),("year","hist_year.png"),("odometer","hist_odometer.png")]:
    plt.figure(figsize=(8,5))
    df[col].dropna().plot(kind="hist", bins=50)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col); plt.ylabel("Count"); plt.tight_layout()
    plt.savefig(f"outputs/eda/{fn}", dpi=150)
    plt.close()

def scatter_with_trend(x, y, out):
    d = df[[x,y]].dropna()
    plt.figure(figsize=(8,5))
    plt.scatter(d[x], d[y], s=8, alpha=0.4)
    # simple linear trend
    coeffs = np.polyfit(d[x], d[y], 1)
    xs = np.linspace(d[x].min(), d[x].max(), 200)
    plt.plot(xs, coeffs[0]*xs + coeffs[1], linewidth=2)
    plt.title(f"{y} vs {x}")
    plt.xlabel(x); plt.ylabel(y); plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

scatter_with_trend("year", "price", "outputs/eda/price_vs_year.png")
scatter_with_trend("odometer", "price", "outputs/eda/price_vs_odometer.png")
```

---

## 3) Prep Data for Modeling

**Target:** `price`  
**Predictors:**  
`year, odometer, manufacturer, condition, cylinders, fuel, title_status, transmission, drive, type, paint_color, state`

Strategy:
- Handle missings:
  - Drop **key numeric nulls** (`price`, `year`, `odometer`).
  - Impute or drop remaining as needed; default to dropping sparse rows for a simple baseline.
- One‑hot encode categoricals.
- Optional **row sampling ~30k** for faster iteration when the dataset is huge.

**Example:**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

df = pd.read_parquet("data/processed/vehicles_clean.parquet").copy()

# Keep only needed columns
features = ["year","odometer","manufacturer","condition","cylinders","fuel",
            "title_status","transmission","drive","type","paint_color","state"]
df = df.dropna(subset=["price","year","odometer"])

# Optional speed-up
if len(df) > 30000:
    df = df.sample(30000, random_state=42)

X = df[features]
y = df["price"]

num_cols = ["year","odometer"]
cat_cols = [c for c in features if c not in num_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num","passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)

model = Pipeline(steps=[("prep", preprocess), ("lr", LinearRegression(n_jobs=None))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
```

---

## 4) Model

**Model:** Linear Regression (with one‑hot encoded categoricals)  
**Metrics:** R² and MAE on a held‑out test set  
**Coefficient Analysis:** Extract and rank positive/negative drivers

**Example (metrics + coefficients):**
```python
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

pred = model.predict(X_test)
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
print({"r2": r2, "mae": mae})

# Get feature names after one-hot
ohe = model.named_steps["prep"].named_transformers_["cat"]
ohe_features = list(ohe.get_feature_names_out(cat_cols))
feature_names = num_cols + ohe_features

coefs = model.named_steps["lr"].coef_
coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs}).sort_values("coef", ascending=False)
coef_df.to_csv("outputs/model/coefficients.csv", index=False)

# Top +/-
top_pos = coef_df.head(15)
top_neg = coef_df.tail(15).sort_values("coef")
top_pos.to_csv("outputs/model/top_positive_coefficients.csv", index=False)
top_neg.to_csv("outputs/model/top_negative_coefficients.csv", index=False)

# Save metrics
import json, os
os.makedirs("outputs/model", exist_ok=True)
with open("outputs/model/metrics.json","w") as f:
    json.dump({"r2": r2, "mae": mae, "n_train": len(X_train), "n_test": len(X_test)}, f, indent=2)
```

**Coefficient bar chart:**
```python
import matplotlib.pyplot as plt

def coef_bar(coef_df, out_path, title):
    plt.figure(figsize=(10,6))
    plt.barh(coef_df["feature"], coef_df["coef"])
    plt.title(title)
    plt.xlabel("Coefficient (price units)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

coef_bar(top_pos.iloc[::-1], "outputs/charts/coef_top_positive.png", "Top Positive Drivers")
coef_bar(top_neg, "outputs/charts/coef_top_negative.png", "Top Negative Drivers")
```

---

## 5) Deliverables

- **EDA plots:** distributions (price, year, odometer) and relationships (Price–Year, Price–Odometer)
- **Coefficient bar charts:** top positive & top negative drivers
- **Single‑page dashboard:** lightweight HTML summary (key metrics + charts)
- **PPT deck:** 4–6 slides (Problem, Data, EDA, Model, Drivers, Next Steps)

**Auto‑generate dashboard & deck (example stubs):**
```python
# dashboard.html (very simple static writer)
from pathlib import Path
import json

metrics = json.loads(Path("outputs/model/metrics.json").read_text())

html = f'''
<!doctype html>
<html><head><meta charset="utf-8"><title>Vehicles Model Dashboard</title></head>
<body style="font-family: sans-serif; max-width: 900px; margin: auto;">
  <h1>Vehicles Model — Dashboard</h1>
  <h2>Metrics</h2>
  <pre>{metrics}</pre>
  <h2>EDA</h2>
  <img src="../eda/hist_price.png" width="400">
  <img src="../eda/hist_year.png" width="400">
  <img src="../eda/hist_odometer.png" width="400">
  <h2>Relationships</h2>
  <img src="../eda/price_vs_year.png" width="450">
  <img src="../eda/price_vs_odometer.png" width="450">
  <h2>Drivers</h2>
  <img src="../charts/coef_top_positive.png" width="450">
  <img src="../charts/coef_top_negative.png" width="450">
</body></html>
'''
Path("outputs/dashboard").mkdir(parents=True, exist_ok=True)
Path("outputs/dashboard/dashboard.html").write_text(html)
print("Wrote outputs/dashboard/dashboard.html")
```

**PowerPoint deck (python‑pptx outline):**
```python
# pip install python-pptx
from pptx import Presentation
from pptx.util import Inches

prs = Presentation()
title_slide_layout = prs.slide_layouts[0]
slide = prs.slides.add_slide(title_slide_layout)
slide.shapes.title.text = "Vehicles Price Modeling — Summary"
slide.placeholders[1].text = "Data • EDA • Linear Regression • Drivers"

# Add slides and images as needed
for title, img in [
    ("Distribution: Price", "outputs/eda/hist_price.png"),
    ("Distribution: Year", "outputs/eda/hist_year.png"),
    ("Distribution: Odometer", "outputs/eda/hist_odometer.png"),
    ("Price vs Year", "outputs/eda/price_vs_year.png"),
    ("Price vs Odometer", "outputs/eda/price_vs_odometer.png"),
    ("Top Positive Drivers", "outputs/charts/coef_top_positive.png"),
    ("Top Negative Drivers", "outputs/charts/coef_top_negative.png"),
]:
    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = title
    s.shapes.add_picture(img, Inches(1), Inches(1.5), width=Inches(8))

prs.save("outputs/deck/Vehicles_Model_Summary.pptx")
print("Saved PPT deck to outputs/deck/Vehicles_Model_Summary.pptx")
```

---

## How to Run

1) **Install**
```
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Minimal `requirements.txt`:
```
pandas
numpy
matplotlib
scikit-learn
python-pptx
pyarrow
```

2) **Place your CSV** at `data/raw/vehicles.csv`

3) **Run the pipeline** (example order):
```
python src/load_and_filter.py
python src/eda.py
python src/prep.py
python src/model.py
python src/report.py

---

## Notes & Gotchas

- The linear model is a **baseline**; it’s fast and interpretable. For higher accuracy, consider regularization (Ridge/Lasso), tree ensembles, or GBMs—keep the same prep steps.
- Watch for **data leakage** if you do any target‑based imputations or filtering post split.
- Consider **winsorizing** extreme prices/odometer after the initial sanity filter.
- For explainability, supplement coefficients with **partial dependence** or **accumulated local effects (ALE)** on numeric features.

---

## Success Criteria Checklist

- [ ] Clean `vehicles_clean.parquet` saved
- [ ] 3 histograms (price/year/odometer)
- [ ] 2 scatter+trend charts
- [ ] Train/test metrics (R², MAE) saved to `metrics.json`
- [ ] Ranked coefficients CSV + top +/- charts
- [ ] `dashboard.html` renders locally
- [ ] `Vehicles_Model_Summary.pptx` exports with key visuals

---

## License

MIT (or your org’s standard).

