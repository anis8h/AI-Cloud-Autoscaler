import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

DATA_FILE = "data/workload.csv"

print("==========================================")
print(" LEAKAGE-FREE AI CLOUD PREDICTION MODEL")
print("==========================================")

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(DATA_FILE)

if len(df) < 10:
    print("\nNot enough data!")
    print("Please collect more monitoring records.")
    exit()

# ==========================================
# CREATE LAG FEATURES
# ==========================================

df["cpu_lag1"] = df["cpu_usage"].shift(1)
df["memory_lag1"] = df["memory_usage"].shift(1)

df["cpu_lag2"] = df["cpu_usage"].shift(2)
df["memory_lag2"] = df["memory_usage"].shift(2)

# Future CPU = target
df["future_cpu"] = df["cpu_usage"].shift(-1)

# Remove missing rows
df = df.dropna()

# ==========================================
# INPUT FEATURES
# ==========================================

X = df[
    [
        "cpu_lag1",
        "memory_lag1",
        "cpu_lag2",
        "memory_lag2"
    ]
]

# Target = NEXT CPU
y = df["future_cpu"]

# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# AI MODEL
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)

# ==========================================
# TEST PREDICTION
# ==========================================

predictions = model.predict(X_test)

# ==========================================
# PERFORMANCE METRICS
# ==========================================

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n------------------------------------------")
print("MODEL PERFORMANCE")
print("------------------------------------------")

print(f"Training Records : {len(X_train)}")
print(f"Testing Records  : {len(X_test)}")

print(f"\nMean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score                 : {r2:.2f}")

# ==========================================
# LATEST DATA FOR FUTURE PREDICTION
# ==========================================

latest_features = pd.DataFrame(
    [[
        df["cpu_usage"].iloc[-1],
        df["memory_usage"].iloc[-1],
        df["cpu_usage"].iloc[-2],
        df["memory_usage"].iloc[-2]
    ]],
    columns=[
        "cpu_lag1",
        "memory_lag1",
        "cpu_lag2",
        "memory_lag2"
    ]
)

next_cpu = model.predict(latest_features)[0]

# Keep prediction within valid CPU range
next_cpu = max(0, min(100, next_cpu))

# ==========================================
# WORKLOAD CLASSIFICATION
# ==========================================

if next_cpu >= 70:
    workload = "HIGH"
elif next_cpu >= 40:
    workload = "MEDIUM"
else:
    workload = "LOW"

# ==========================================
# CURRENT STATUS
# ==========================================

current_cpu = df["cpu_usage"].iloc[-1]
current_memory = df["memory_usage"].iloc[-1]

print("\n------------------------------------------")
print("CURRENT RESOURCE STATUS")
print("------------------------------------------")

print(f"Current CPU Usage    : {current_cpu:.2f}%")
print(f"Current Memory Usage : {current_memory:.2f}%")

print("\n------------------------------------------")
print("AI NEXT-CPU PREDICTION")
print("------------------------------------------")

print(f"Predicted Next CPU   : {next_cpu:.2f}%")
print(f"Predicted Workload   : {workload}")

print("\n==========================================")
print(" LEAKAGE-FREE AI MODEL TEST COMPLETED")
print("==========================================")