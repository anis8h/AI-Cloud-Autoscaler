
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor

DATA_FILE = Path("data/workload.csv")

print("=" * 50)
print(" AI MODEL IMPROVEMENT - EVALUATION")
print("=" * 50)

if not DATA_FILE.exists():
    raise FileNotFoundError("data/workload.csv not found. Run from project folder.")

df = pd.read_csv(DATA_FILE)

for col in ["cpu_usage", "memory_usage"]:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["cpu_usage", "memory_usage"]).reset_index(drop=True)

# Features at time t predict CPU at time t+1
df["previous_cpu"] = df["cpu_usage"].shift(1)
df["previous_memory"] = df["memory_usage"].shift(1)
df["cpu_avg_3"] = df["cpu_usage"].rolling(3).mean()
df["memory_avg_3"] = df["memory_usage"].rolling(3).mean()
df["future_cpu"] = df["cpu_usage"].shift(-1)

feature_columns = [
    "cpu_usage",
    "memory_usage",
    "previous_cpu",
    "previous_memory",
    "cpu_avg_3",
    "memory_avg_3",
]

data = df.dropna(subset=feature_columns + ["future_cpu"]).reset_index(drop=True)

if len(data) < 15:
    raise ValueError("Not enough data. Collect more readings before evaluation.")

X = data[feature_columns]
y = data["future_cpu"]

# Chronological split; do not shuffle time-series data
split = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Baseline predicts next CPU using the current CPU reading
baseline_predictions = X_test["cpu_usage"].to_numpy()

# Compare Linear Regression with a simple mean-prediction baseline
mean_baseline = DummyRegressor(strategy="mean")
mean_baseline.fit(X_train, y_train)
mean_predictions = mean_baseline.predict(X_test)


# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
ai_predictions = model.predict(X_test)

# Random Forest: small model for a low-RAM computer
rf_model = RandomForestRegressor(
    n_estimators=50,
    max_depth=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=1
)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)


baseline_mae = mean_absolute_error(y_test, baseline_predictions)
mean_mae = mean_absolute_error(y_test, mean_predictions)
ai_mae = mean_absolute_error(y_test, ai_predictions)
ai_r2 = r2_score(y_test, ai_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print("\nDATASET")
print("-" * 50)
print(f"Valid records       : {len(data)}")
print(f"Training records    : {len(X_train)}")
print(f"Testing records     : {len(X_test)}")
print("Split               : Chronological (80/20)")

print("\nMODEL COMPARISON")
print("-" * 50)
print(f"Current CPU baseline MAE : {baseline_mae:.2f}")
print(f"Mean baseline MAE        : {mean_mae:.2f}")
print(f"Linear Regression MAE    : {ai_mae:.2f}")
print(f"Linear Regression R2     : {ai_r2:.2f}")
print(f"Random Forest MAE       : {rf_mae:.2f}")
print(f"Random Forest R2        : {rf_r2:.2f}")

best_baseline = min(baseline_mae, mean_mae)
if ai_mae < best_baseline:
    print("Result: Linear Regression beats both baselines on MAE.")
else:
    print("Result: Linear Regression has not beaten both baselines yet.")

# Build live features using exactly the same feature definitions
latest = df.iloc[-1]
latest_previous = df.iloc[-2]
recent = df.tail(3)

live_features = pd.DataFrame([{
    "cpu_usage": latest["cpu_usage"],
    "memory_usage": latest["memory_usage"],
    "previous_cpu": latest_previous["cpu_usage"],
    "previous_memory": latest_previous["memory_usage"],
    "cpu_avg_3": recent["cpu_usage"].mean(),
    "memory_avg_3": recent["memory_usage"].mean(),
}])[feature_columns]

predicted_cpu = float(model.predict(live_features)[0])
predicted_cpu = max(0.0, min(100.0, predicted_cpu))

if predicted_cpu < 40:
    workload = "LOW"
elif predicted_cpu < 70:
    workload = "MEDIUM"
else:
    workload = "HIGH"

print("\nCURRENT RESOURCE STATUS")
print("-" * 50)
print(f"Current CPU         : {latest['cpu_usage']:.2f}%")
print(f"Current Memory      : {latest['memory_usage']:.2f}%")
print(f"Predicted Next CPU  : {predicted_cpu:.2f}%")
print(f"Predicted Workload  : {workload}")
print("\nEvaluation completed.")
