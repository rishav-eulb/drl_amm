import requests
import time
import csv

API = "https://api.hyperliquid.xyz/info"
COIN = "HYPE"
INTERVAL = "1m"
MS = 1000

# 200 days in minutes
TARGET_CANDLES = 200 * 24 * 60

end_ms = int(time.time() * MS)
all_rows = []

def fetch(end_time):
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": COIN,
            "interval": INTERVAL,
            "startTime": 0,
            "endTime": end_time
        }
    }
    r = requests.post(API, json=payload)
    r.raise_for_status()
    return r.json()

while len(all_rows) < TARGET_CANDLES:
    data = fetch(end_ms)

    if not data:
        break

    print(f"Fetched {len(data)}")

    all_rows = data + all_rows   # prepend because data is most recent → oldest

    # move end_ms earlier:
    earliest_open = data[0]["t"]
    end_ms = earliest_open - 1

print(f"Total fetched = {len(all_rows)}")

# Trim exactly 200 days
all_rows = all_rows[-TARGET_CANDLES:]

# Write file
with open("HYPE_1m_200days.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(["openTime","closeTime","open","high","low","close","volume"])
    for r in all_rows:
        w.writerow([r["t"], r["T"], r["o"], r["h"], r["l"], r["c"], r["v"]])

print("DONE")
