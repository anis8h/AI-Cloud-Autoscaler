# AI-Based Intelligent Cloud Resource Allocation and Auto-Scaling for Cost Optimization

## 1. Project Overview

This project is an AI-assisted cloud resource monitoring and auto-scaling simulation system designed to predict upcoming CPU utilization, monitor memory usage, make scaling decisions, and estimate infrastructure costs.

It uses a Machine Learning model based on Linear Regression to predict the next CPU usage value. A rule-based scaling policy combines the predicted CPU usage with current CPU and memory utilization to recommend whether to scale up, maintain, or scale down the number of instances.

The project provides a Streamlit dashboard to visualize resource metrics, predictions, scaling decisions, and estimated costs.

## 2. Objectives

- Monitor CPU and memory utilization.
- Predict upcoming CPU usage using Machine Learning.
- Implement intelligent scaling decisions.
- Estimate hourly infrastructure costs.
- Record scaling decisions in a CSV log.
- Display resource and cost information through an interactive dashboard.

## 3. Key Features

- CPU and memory monitoring
- Linear Regression-based CPU prediction
- Intelligent scale-up, maintain, and scale-down decisions
- Simulated cost and savings calculations
- CSV-based scaling event logging
- Streamlit dashboard
- Model evaluation using MAE and R²
- Automated tests for scaling policy and cost calculations

## 4. Technology Stack

- Programming Language: Python
- Machine Learning: Scikit-learn, Linear Regression
- Data Processing: Pandas
- Dashboard: Streamlit
- Data Storage: CSV files
- Development Environment: Visual Studio Code
- Version Control: Git and GitHub

## 5. Project Architecture

The system follows this workflow:

1. Resource monitoring collects CPU and memory utilization.
2. Workload data is processed to generate model features.
3. The Linear Regression model predicts the next CPU utilization.
4. The scaling policy evaluates predicted CPU, current CPU, and current memory.
5. The system recommends SCALE UP, MAINTAIN, or SCALE DOWN.
6. The cost optimizer estimates the hourly cost before and after the decision.
7. Scaling events are stored in a CSV file.
8. The Streamlit dashboard displays the results.

## 6. Scaling Policy

The current simulated policy uses these conditions:

- SCALE UP: Predicted CPU is at least 70%, current CPU is at least 80%, or current memory is at least 85%.
- MAINTAIN: If the scale-up condition is false, predicted CPU is at least 40%, current CPU is at least 40%, or current memory is at least 70%.
- SCALE DOWN: If neither of the above conditions is met.

The instance count is adjusted by one in the selected direction, with a minimum of one instance.

## 7. Cost Calculation

The project uses a configurable simulated cost of ₹5 per instance per hour.

**Formula:**

Cost Before = Instances Before × Cost per Instance

Cost After = Instances After × Cost per Instance

Cost Saving = Cost Before − Cost After

Savings Percentage = (Cost Saving / Cost Before) × 100

A negative saving indicates an increase in estimated cost. Scaling up may increase cost while helping address high resource utilization.

## 8. Project Structure

```text
AI-Cloud-Autoscaler/
├── ai/
│   ├── model_evaluation.py
│   └── predictor.py
├── autoscaling/
│   ├── cost_optimizer.py
│   └── scaler.py
├── dashboard/
│   ├── dashboard.py
│   └── dashboard_backup.py
├── data/
│   ├── workload.csv
│   └── scaling_events.csv
├── monitoring/
│   └── monitor.py
├── test_scaling_policy.py
├── test_cost_calculation.py
└── README.md
```

## 9. Installation and Setup

### Prerequisites

- Python 3
- Visual Studio Code
- pip

### Step 1: Open the project folder

```powershell
cd C:\Users\Shree\AI-Cloud-Autoscaler
```

### Step 2: Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
python -m pip install pandas scikit-learn streamlit streamlit-autorefresh
```

## 10. Running the Project

### Launch the dashboard

```powershell
python -m streamlit run dashboard\dashboard.py
```

Open the local URL displayed in the terminal, usually:

http://localhost:8501

### Run the scaling engine

```powershell
python autoscaling\scaler.py
```

### Run the cost optimizer

```powershell
python autoscaling\cost_optimizer.py
```

### Evaluate the Machine Learning model

```powershell
python ai\model_evaluation.py
```

### Run the scaling policy tests

```powershell
python test_scaling_policy.py
```

### Run the cost calculation tests

```powershell
python test_cost_calculation.py
```

## 11. Sample Output

An example execution produced the following values:

- Current CPU Usage: 23.80%
- Current Memory Usage: 88.00%
- Predicted Next CPU Usage: 27.79%
- Predicted Workload: LOW
- Scaling Decision: SCALE UP
- Instances Before: 2
- Instances After: 3
- Estimated Cost Before: ₹10/hour
- Estimated Cost After: ₹15/hour
- Estimated Cost Change: −₹5/hour in savings

The scale-up decision occurs because memory utilization is above the configured 85% threshold, even though predicted CPU utilization is low.

These values are an example of one execution and may change with new workload data.

## 12. Testing and Evaluation

The scaling policy tests verify the expected decisions for scale-up, maintain, and scale-down scenarios.

The cost calculation tests verify the estimated cost and savings calculations for all three scenarios.

The Machine Learning evaluation script compares Linear Regression performance against baseline predictions using Mean Absolute Error (MAE) and R².

In the current evaluation, Linear Regression has not outperformed both baseline approaches. Therefore, its predictive performance requires further improvement and should not be considered superior to the baselines.

## 13. Limitations

- This project is a simulation and does not create, remove, or scale real cloud instances.
- The hourly cost is an illustrative value, not actual cloud-provider pricing.
- The Machine Learning model's accuracy depends on the available workload data.
- The current scaling policy is rule-based and uses fixed thresholds.
- Further testing with larger, more diverse workloads is needed before real-world use.

## 14. Future Enhancements

- Integrate a cloud provider such as AWS or Azure.
- Add real-time cloud metrics collection.
- Improve prediction accuracy using more workload data and alternative models.
- Add cloud-provider-specific cost estimates.
- Implement alerts, logging improvements, and model retraining.
- Evaluate scaling performance under simulated workload spikes.

## 15. Conclusion

This project demonstrates how Machine Learning predictions, resource monitoring, rule-based auto-scaling, cost estimation, and dashboard visualization can be combined in a cloud resource optimization workflow.

It provides a foundation for further study and future integration with real cloud infrastructure.
