import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Cloud Resource Optimizer",
    page_icon="☁️",
    layout="wide"
)

# Auto refresh every 5 seconds
st_autorefresh(interval=5000, key="cloud_monitor_refresh")


# ==========================================
# FILES
# ==========================================

DATA_FILE = "data/workload.csv"
LOG_FILE = "data/scaling_events.csv"

COST_PER_INSTANCE = 5.0
CURRENT_INSTANCES = 2


# ==========================================
# TITLE
# ==========================================

st.title("☁️ AI-Based Intelligent Cloud Resource Allocation")
st.subheader("Auto-Scaling and Cost Optimization Dashboard")

st.info(
    "Dashboard automatically refreshes every 5 seconds "
    "using the latest monitoring data."
)


# ==========================================
# LOAD MONITORING DATA
# ==========================================

try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    st.error("workload.csv not found. Please run monitor.py first.")
    st.stop()


if len(df) < 10:
    st.warning(
        f"Not enough monitoring data. Current records: {len(df)}. "
        "Please collect at least 10 records."
    )
    st.stop()


# ==========================================
# AI MODEL - LEAKAGE FREE
# ==========================================

df["cpu_lag1"] = df["cpu_usage"].shift(1)
df["memory_lag1"] = df["memory_usage"].shift(1)

df["cpu_lag2"] = df["cpu_usage"].shift(2)
df["memory_lag2"] = df["memory_usage"].shift(2)


# Features at time t to predict CPU at time t+1
df["previous_cpu"] = df["cpu_usage"].shift(1)
df["previous_memory"] = df["memory_usage"].shift(1)

df["cpu_avg_3"] = df["cpu_usage"].rolling(3).mean()
df["memory_avg_3"] = df["memory_usage"].rolling(3).mean()

# Target = NEXT CPU
df["future_cpu"] = df["cpu_usage"].shift(-1)

model_df = df.dropna(
    subset=[
        "cpu_usage",
        "memory_usage",
        "previous_cpu",
        "previous_memory",
        "cpu_avg_3",
        "memory_avg_3",
        "future_cpu",
    ]
).copy()

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

# ==========================================
# TRAIN MODEL
# ==========================================

model = LinearRegression()

model.fit(X, y)


# ==========================================
# CURRENT RESOURCE STATUS
# ==========================================

current_cpu = df["cpu_usage"].iloc[-1]
current_memory = df["memory_usage"].iloc[-1]


# ==========================================
# PREDICT NEXT CPU
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


# Prepare latest resource features for prediction
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

predicted_cpu = float(model.predict(latest_features)[0])

# Keep prediction within valid CPU range
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


if (
    predicted_cpu >= 70
    or current_cpu >= 80
    or current_memory >= 85
):
    decision = "SCALE UP"
    required_instances = CURRENT_INSTANCES + 1

elif (
    predicted_cpu >= 40
    or current_cpu >= 40
    or current_memory >= 70
):
    decision = "MAINTAIN"
    required_instances = CURRENT_INSTANCES

else:
    decision = "SCALE DOWN"
    required_instances = max(
        1,
        CURRENT_INSTANCES - 1
    )

# ==========================================
# COST CALCULATION
# ==========================================

cost_before = CURRENT_INSTANCES * COST_PER_INSTANCE

cost_after = required_instances * COST_PER_INSTANCE

savings = cost_before - cost_after

if cost_before > 0:
    savings_percentage = (
        savings / cost_before
    ) * 100
else:
    savings_percentage = 0


# ==========================================
# RESOURCE DASHBOARD
# ==========================================

st.markdown("## 📊 Current Resource Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "CPU Usage",
        f"{current_cpu:.2f}%"
    )

with col2:
    st.metric(
        "Memory Usage",
        f"{current_memory:.2f}%"
    )

with col3:
    st.metric(
        "Predicted Next CPU",
        f"{predicted_cpu:.2f}%"
    )


# ==========================================
# AI PREDICTION
# ==========================================

st.markdown("## 🤖 AI Prediction")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Predicted Workload",
        workload
    )

with col2:
    st.metric(
        "Prediction Target",
        "Next CPU Usage"
    )


# ==========================================
# AUTO SCALING
# ==========================================

st.markdown("## ⚙️ Intelligent Auto-Scaling")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Decision",
        decision
    )

with col2:
    st.metric(
        "Instances Before",
        CURRENT_INSTANCES
    )

with col3:
    st.metric(
        "Instances After",
        required_instances
    )


# ==========================================
# COST OPTIMIZATION
# ==========================================

st.markdown("## 💰 Cost Optimization")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Cost Before AI",
        f"₹{cost_before:.2f}/hr"
    )

with col2:
    st.metric(
        "Cost After AI",
        f"₹{cost_after:.2f}/hr"
    )

with col3:
    st.metric(
        "Cost Saving",
        f"₹{savings:.2f}/hr"
    )

with col4:
    st.metric(
        "Saving %",
        f"{savings_percentage:.2f}%"
    )

st.caption(
    "Note: ₹5.00 per instance/hour is a simulated project cost, "
    "not actual AWS/Azure/GCP pricing."
)


# ==========================================
# RESOURCE USAGE GRAPH
# ==========================================

st.markdown("## 📈 Resource Usage History")

chart_data = df[
    ["cpu_usage", "memory_usage"]
].tail(30)

st.line_chart(chart_data)


# ==========================================
# SCALING POLICY
# ==========================================


st.markdown("## 📋 Scaling Policy")

policy_data = pd.DataFrame({
    "Condition": [
        "Predicted CPU >= 70% OR Current CPU >= 80% OR Memory >= 85%",
        "Predicted CPU >= 40% OR Current CPU >= 40% OR Memory >= 70%",
        "CPU and Memory usage are below scaling thresholds"
    ],
    "Action": [
        "SCALE UP",
        "MAINTAIN",
        "SCALE DOWN"
    ]
})

st.table(policy_data)


# ==========================================
# SCALING EVENT HISTORY
# ==========================================

st.markdown("## 📝 Scaling Event History")

try:

    log_df = pd.read_csv(LOG_FILE)

    if len(log_df) > 0:

        st.dataframe(
            log_df.tail(10),
            width="stretch"
        )

    else:

        st.info("No scaling events recorded yet.")

except FileNotFoundError:

    st.info(
        "No scaling event history available yet. "
        "Run cost_optimizer.py first."
    )


# ==========================================
# SYSTEM INFORMATION
# ==========================================

st.markdown("---")

st.caption(
    "AI-Based Intelligent Cloud Resource Allocation "
    "and Auto-Scaling for Cost Optimization"
)

st.caption(
    "Model: Linear Regression | "
    "Features: CPU/Memory Lag-1 and Lag-2 | "
    "Target: Next CPU Usage"
)