import re
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


class monitor:
    def __init__(self, log_file: str):
        self.log_file = log_file

    # Parse log entries
    def parse_log(self):
        log_data = []
        try:
            with open(self.log_file, "r") as f:
                lines = f.readlines()
                print(f"📄 Reading {len(lines)} lines from {self.log_file}")

                for line in lines:
                    # Extract timestamp and JSON data
                    match = re.search(
                        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ INFO ({.+})", line
                    )
                    if match:
                        timestamp_str = match.group(1)
                        json_str = match.group(2)
                        try:
                            data = json.loads(json_str)
                            data["timestamp"] = datetime.strptime(
                                timestamp_str, "%Y-%m-%d %H:%M:%S"
                            )
                            log_data.append(data)
                        except json.JSONDecodeError as e:
                            print(
                                f"⚠️ Failed to parse JSON: {json_str[:50]}... Error: {e}"
                            )
                            continue
                    else:
                        print(f"⚠️ Line didn't match pattern: {line[:80]}...")

                print(f"✅ Successfully parsed {len(log_data)} log entries")
        except FileNotFoundError:
            print(f"❌ File not found: {self.log_file}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")

        return log_data

    def plot_stats(self):
        log_data = self.parse_log()

        # Create DataFrame
        df = pd.DataFrame(log_data)

        if len(df) > 0:
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("API Prediction Stats Over Time", fontsize=16)

            # Plot 1: Latency over time
            axes[0, 0].plot(
                df["timestamp"], df["latency"], marker="o", linestyle="-", color="blue"
            )
            axes[0, 0].set_title("Prediction Latency")
            axes[0, 0].set_xlabel("Time")
            axes[0, 0].set_ylabel("Latency (seconds)")
            axes[0, 0].tick_params(axis="x", rotation=45)
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Mean input values
            axes[0, 1].plot(
                df["timestamp"], df["mean"], marker="s", linestyle="-", color="green"
            )
            axes[0, 1].set_title("Mean Input Value")
            axes[0, 1].set_xlabel("Time")
            axes[0, 1].set_ylabel("Mean")
            axes[0, 1].tick_params(axis="x", rotation=45)
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Standard deviation
            axes[1, 0].plot(
                df["timestamp"], df["std"], marker="^", linestyle="-", color="orange"
            )
            axes[1, 0].set_title("Input Standard Deviation")
            axes[1, 0].set_xlabel("Time")
            axes[1, 0].set_ylabel("Std Dev")
            axes[1, 0].tick_params(axis="x", rotation=45)
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Request count
            axes[1, 1].plot(
                df["timestamp"], df["count"], marker="d", linestyle="-", color="red"
            )
            axes[1, 1].set_title("Instances per Request")
            axes[1, 1].set_xlabel("Time")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].tick_params(axis="x", rotation=45)
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
        else:
            print(f"❌ No valid log entries found in {self.log_file}")

    def display_summary(self):
        log_data = self.parse_log()
        df = pd.DataFrame(log_data)

        if not df.empty:
            # Display summary statistics
            print("\n📊 Summary Statistics:")
            print(f"Total requests: {len(df)}")
            print(f"Avg latency: {df['latency'].mean():.6f}s")
            print(f"Min latency: {df['latency'].min():.6f}s")
            print(f"Max latency: {df['latency'].max():.6f}s")
            print(f"\nLatest predictions: {df.iloc[-1]['preds']}")
        else:
            print(f"❌ No log data found in {self.log_file}")
