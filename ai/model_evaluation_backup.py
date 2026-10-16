import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATA_FILE = "data/workload.csv"

print("==========================================")
print(" AI MODEL PERFORMANCE EVALUATION")
print("==========================================")

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(DATA_FILE)

if len(df) < 15:
    print("\nNot enough data for evaluation.")
    print("Please collect at least 15 monitoring records.")
    exit()

# ==========================================
# CREATE LAG FEATURES
# ==========================================

df["cpu_lag1"] = df["cpu_usage"].shift(1)
df["memory_lag1"] = df["memory_usage"].shift(1)

df["cpu_lag2"] = df["cpu_usage"].shift(2)
df["memory_lag2"] = df["memory_usage"].shift(2)

# Target = next CPU usage
df["future_cpu"] = df["cpu_usage"].shift(-1)

df = df.dropna()

# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df[
    [
        "cpu_lag1",
        "memory_lag1",
        "cpu_lag2",
        "memory_lag2"
    ]
]

y = df["future_cpu"]

# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# TRAIN MODEL
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)

# ==========================================
# PREDICTION
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

# ==========================================
# CURRENT PREDICTION
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

current_cpu = df["cpu_usage"].iloc[-1]
current_memory = df["memory_usage"].iloc[-1]

next_cpu = model.predict(
    latest_features
)[0]

next_cpu = max(
    0,
    min(100, next_cpu)
)

# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n------------------------------------------")
print("DATASET")
print("------------------------------------------")

print(f"Total Records       : {len(df)}")
print(f"Training Records    : {len(X_train)}")
print(f"Testing Records     : {len(X_test)}")

print("\n------------------------------------------")
print("MODEL PERFORMANCE")
print("------------------------------------------")

print(f"MAE                 : {mae:.2f}")
print(f"R² Score            : {r2:.2f}")

print("\n------------------------------------------")
print("CURRENT RESOURCE STATUS")
print("------------------------------------------")

print(f"Current CPU         : {current_cpu:.2f}%")
print(f"Current Memory      : {current_memory:.2f}%")

print("\n------------------------------------------")
print("AI PREDICTION")
print("------------------------------------------")

print(f"Predicted Next CPU  : {next_cpu:.2f}%")

if next_cpu >= 70:
    workload = "HIGH"
elif next_cpu >= 40:
    workload = "MEDIUM"
else:
    workload = "LOW"

print(f"Predicted Workload  : {workload}")

print("\n==========================================")
print(" MODEL EVALUATION COMPLETED")
print("==========================================")