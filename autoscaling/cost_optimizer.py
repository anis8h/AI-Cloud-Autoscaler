import pandas as pd
import csv
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression

DATA_FILE = "data/workload.csv"
LOG_FILE = "data/scaling_events.csv"

# Simulated cloud cost
COST_PER_INSTANCE = 5.0

print("==========================================")
print(" AI CLOUD COST OPTIMIZATION ENGINE")
print("==========================================")

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(DATA_FILE)

if len(df) < 10:
    print("\nNot enough data for cost optimization.")
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
# CURRENT FEATURES
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
# AI NEXT CPU PREDICTION
# ==========================================

predicted_cpu = float(model.predict(latest_features)[0])

# Keep CPU prediction within valid range
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
# AUTO-SCALING POLICY
# ==========================================

current_instances = 2

# SCALE UP: High CPU or memory pressure
if (
    predicted_cpu >= 70
    or current_cpu >= 80
    or current_memory >= 85
):
    decision = "SCALE UP"
    required_instances = current_instances + 1

# MAINTAIN: Moderate resource usage
elif (
    predicted_cpu >= 40
    or current_cpu >= 40
    or current_memory >= 70
):
    decision = "MAINTAIN"
    required_instances = current_instances

# SCALE DOWN: Resources are low
else:
    decision = "SCALE DOWN"
    required_instances = max(1, current_instances - 1)

# ==========================================
# COST CALCULATION
# ==========================================

cost_before = (
    current_instances *
    COST_PER_INSTANCE
)

cost_after = (
    required_instances *
    COST_PER_INSTANCE
)

savings = cost_before - cost_after

if cost_before > 0:

    savings_percentage = (
        savings /
        cost_before
    ) * 100

else:

    savings_percentage = 0

# ==========================================
# DISPLAY RESOURCE STATUS
# ==========================================

print("\n------------------------------------------")
print("CURRENT RESOURCE STATUS")
print("------------------------------------------")

print(f"Current CPU        : {current_cpu:.2f}%")
print(f"Current Memory     : {current_memory:.2f}%")

# ==========================================
# DISPLAY AI PREDICTION
# ==========================================

print("\n------------------------------------------")
print("AI PREDICTION")
print("------------------------------------------")

print(f"Predicted Next CPU : {predicted_cpu:.2f}%")
print(f"Predicted Workload : {workload}")

# ==========================================
# DISPLAY SCALING
# ==========================================

print("\n------------------------------------------")
print("AUTO-SCALING")
print("------------------------------------------")

print(f"Decision            : {decision}")
print(f"Instances Before   : {current_instances}")
print(f"Instances After    : {required_instances}")

# ==========================================
# DISPLAY COST
# ==========================================

print("\n------------------------------------------")
print("COST OPTIMIZATION")
print("------------------------------------------")

print(f"Cost Before AI     : ₹{cost_before:.2f}/hour")
print(f"Cost After AI      : ₹{cost_after:.2f}/hour")
print(f"Cost Saving        : ₹{savings:.2f}/hour")
print(f"Savings Percentage : {savings_percentage:.2f}%")

# ==========================================
# CREATE LOG DIRECTORY
# ==========================================

os.makedirs(
    "data",
    exist_ok=True
)

# ==========================================
# CREATE EVENT LOG
# ==========================================

if not os.path.exists(LOG_FILE):

    with open(
        LOG_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "cpu_usage",
            "memory_usage",
            "predicted_cpu",
            "workload",
            "decision",
            "instances_before",
            "instances_after",
            "cost_before",
            "cost_after",
            "savings",
            "savings_percentage"
        ])

# ==========================================
# SAVE SCALING EVENT
# ==========================================

with open(
    LOG_FILE,
    "a",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        round(current_cpu, 2),
        round(current_memory, 2),
        round(predicted_cpu, 2),
        workload,
        decision,
        current_instances,
        required_instances,
        round(cost_before, 2),
        round(cost_after, 2),
        round(savings, 2),
        round(savings_percentage, 2)
    ])

print("\n------------------------------------------")
print("Scaling event saved successfully.")
print(f"Log file: {LOG_FILE}")
print("------------------------------------------")

print("\n==========================================")
print(" AI COST OPTIMIZATION COMPLETED")
print("==========================================")