import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_FILE = "data/workload.csv"

print("==========================================")
print(" AI INTELLIGENT AUTO-SCALING ENGINE")
print("==========================================")

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(DATA_FILE)

if len(df) < 10:
    print("\nNot enough data for auto-scaling.")
    print("Please collect at least 10 monitoring records.")
    exit()


# ==========================================
# CREATE FEATURES
# ==========================================

df["previous_cpu"] = df["cpu_usage"].shift(1)
df["previous_memory"] = df["memory_usage"].shift(1)

df["cpu_avg_3"] = df["cpu_usage"].rolling(3).mean()
df["memory_avg_3"] = df["memory_usage"].rolling(3).mean()

# Target = next CPU
df["future_cpu"] = df["cpu_usage"].shift(-1)

model_df = df.dropna().copy()

# ==========================================
# AI MODEL
# ==========================================

X = model_df[
    [
        "cpu_usage",
        "memory_usage",
        "previous_cpu",
        "previous_memory",
        "cpu_avg_3",
        "memory_avg_3",
    ]
]

y = model_df["future_cpu"]

model = LinearRegression()
model.fit(X, y)



# ==========================================
# CURRENT RESOURCE STATUS
# ==========================================

current_cpu = float(df["cpu_usage"].iloc[-1])
current_memory = float(df["memory_usage"].iloc[-1])

# ==========================================
# PREPARE CURRENT FEATURES
# ==========================================

latest = df.iloc[-1]
previous = df.iloc[-2]
recent = df.tail(3)

latest_features = pd.DataFrame([{
    "cpu_usage": latest["cpu_usage"],
    "memory_usage": latest["memory_usage"],
    "previous_cpu": previous["cpu_usage"],
    "previous_memory": previous["memory_usage"],
    "cpu_avg_3": recent["cpu_usage"].mean(),
    "memory_avg_3": recent["memory_usage"].mean(),
}])[X.columns]

# ==========================================
# PREDICT NEXT CPU
# ==========================================

predicted_cpu = float(model.predict(latest_features)[0])

# Keep prediction between 0 and 100
predicted_cpu = max(0.0, min(100.0, predicted_cpu))


# ==========================================
# WORKLOAD CLASSIFICATION
# ==========================================

if predicted_cpu >= 70:
    workload = "HIGH"

elif predicted_cpu >= 40:
    workload = "MEDIUM"

else:
    workload = "LOW"


# ==========================================
# AUTO-SCALING LOGIC
# ==========================================

current_instances = 2

# SCALE UP: CPU or Memory pressure is high
if (
    predicted_cpu >= 70
    or current_cpu >= 80
    or current_memory >= 85
):
    decision = "SCALE UP"
    required_instances = current_instances + 1

# MAINTAIN: Resource usage is moderate
elif (
    predicted_cpu >= 40
    or current_cpu >= 40
    or current_memory >= 70
):
    decision = "MAINTAIN"
    required_instances = current_instances

# SCALE DOWN: Only when CPU and Memory usage are low
else:
    decision = "SCALE DOWN"
    required_instances = max(1, current_instances - 1)

# ==========================================
# OUTPUT
# ==========================================

print("\n------------------------------------------")
print("CURRENT RESOURCE STATUS")
print("------------------------------------------")

print(f"Current CPU Usage    : {current_cpu:.2f}%")
print(f"Current Memory Usage : {current_memory:.2f}%")

print("\n------------------------------------------")
print("AI PREDICTION")
print("------------------------------------------")

print(f"Predicted Next CPU   : {predicted_cpu:.2f}%")
print(f"Predicted Workload   : {workload}")

print("\n------------------------------------------")
print("AUTO-SCALING DECISION")
print("------------------------------------------")

print(f"Decision             : {decision}")
print(f"Current Instances    : {current_instances}")
print(f"Required Instances   : {required_instances}")

print("\n------------------------------------------")
print("SCALING POLICY")
print("------------------------------------------")

print("SCALE UP   : Predicted CPU >= 70% OR Current CPU >= 80% OR Memory >= 85%")
print("MAINTAIN   : Predicted CPU >= 40% OR Current CPU >= 40% OR Memory >= 70%")
print("SCALE DOWN : CPU and Memory usage are both below their thresholds")

print("\n==========================================")
print(" AI AUTO-SCALING COMPLETED")
print("==========================================")