import random
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# -----------------------
# Configuration
# -----------------------
NUM_RECORDS = 500000
ANOMALY_RATIO = 0.1
START_TIME = datetime.now()

# -----------------------
# URL Pools
# -----------------------
normal_urls = [
    "/home", "/login", "/api/products", "/api/cart",
    "/api/profile", "/search"
]

attack_urls = [
    "/api/login?user=admin'--",
    "/search?q=<script>alert(1)</script>",
    "/../../etc/passwd",
    "/api/login?user=admin OR 1=1"
]

# -----------------------
# User Agents
# -----------------------
user_agents = [
    "Mozilla/5.0",
    "Chrome/120.0",
    "Safari/537.36",
    "curl/7.68.0",
    "python-requests/2.28"
]

# -----------------------
# Helper Functions
# -----------------------
def random_ip():
    return f"192.168.{random.randint(0, 5)}.{random.randint(1, 255)}"

def payload_entropy(is_attack):
    return round(
        random.uniform(4.5, 6.5) if is_attack else random.uniform(1.5, 3.5),
        2
    )

# -----------------------
# HTTP Method Distribution
# -----------------------
NORMAL_METHODS = ["GET", "POST", "PUT", "DELETE"]
NORMAL_WEIGHTS = [0.60, 0.30, 0.06, 0.04]

ATTACK_METHODS = ["POST", "PUT", "DELETE"]
ATTACK_WEIGHTS = [0.60, 0.25, 0.15]

# -----------------------
# Data Generation
# -----------------------
data = []
ip_last_time = {}

for i in range(NUM_RECORDS):

    is_attack = random.random() < ANOMALY_RATIO
    src_ip = random_ip()

    # ⏱ Timestamp simulation
    timestamp = START_TIME + timedelta(seconds=i * random.uniform(0.1, 2))

    last_time = ip_last_time.get(src_ip, timestamp - timedelta(seconds=5))
    time_gap = (timestamp - last_time).total_seconds()
    ip_last_time[src_ip] = timestamp

    # 🌐 HTTP Method selection
    if is_attack:
        method = random.choices(ATTACK_METHODS, ATTACK_WEIGHTS, k=1)[0]
        url = random.choice(attack_urls)
    else:
        method = random.choices(NORMAL_METHODS, NORMAL_WEIGHTS, k=1)[0]
        url = random.choice(normal_urls)

    # 🧾 Record
    row = {
        "timestamp": timestamp.isoformat(),
        "src_ip": src_ip,
        "method": method,
        "url": url,
        "url_length": len(url),
        "query_length": url.count("=") * random.randint(5, 20),
        "status_code": random.choice([401, 403, 500]) if is_attack else 200,
        "bytes_sent": random.randint(3000, 9000) if is_attack else random.randint(200, 2000),
        "user_agent": random.choice(user_agents),
        "request_time": round(
            random.uniform(0.5, 2.5) if is_attack else random.uniform(0.05, 0.4),
            3
        ),
        "req_per_ip_1min": random.randint(100, 300) if is_attack else random.randint(1, 30),
        "req_per_ip_10sec": random.randint(30, 80) if is_attack else random.randint(1, 8),
        "unique_urls_per_ip": random.randint(10, 30) if is_attack else random.randint(1, 5),
        "time_gap": round(time_gap, 3),
        "is_https": random.choice([0, 1]),
        "payload_entropy": payload_entropy(is_attack),
        "label": 1 if is_attack else 0
    }

    data.append(row)

# -----------------------
# Save Dataset
# -----------------------
df = pd.DataFrame(data)
df.to_csv("waf_http_anomaly_dataset.csv", index=False)

# -----------------------
# Verification
# -----------------------
print("✅ Dataset created: waf_http_anomaly_dataset.csv\n")
print("HTTP Method Distribution (%):")
print(df["method"].value_counts(normalize=True) * 100)

print("\nSample Records:")
print(df.head())