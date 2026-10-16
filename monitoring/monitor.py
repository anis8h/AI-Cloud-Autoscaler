import psutil
import csv
import time
import os
from datetime import datetime

# Data folder path
DATA_FILE = "data/workload.csv"

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Create CSV file with headers
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp",
            "cpu_usage",
            "memory_usage"
        ])

print("========================================")
print(" AI Cloud Resource Monitoring Started")
print("========================================")
print("Press CTRL+C to stop\n")

try:
    while True:
        # Get CPU and RAM usage
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().percent

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Display data
        print(
            f"Time: {timestamp} | "
            f"CPU: {cpu}% | "
            f"RAM: {memory}%"
        )

        # Save data
        with open(DATA_FILE, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp,
                cpu,
                memory
            ])

        time.sleep(2)

except KeyboardInterrupt:
    print("\nMonitoring stopped.")