# Copyright (c) 2025 Tucker Balch
#
# Generation of this code was assisted by GitHub Copilot version 1.300.0

"""
This script plots groups of lobster files. Different colors according to UP, DN, or LVL.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import os

def read_lobster_csv(filename):
    times = []
    prices = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            # Only plot execution events (event_type == 4)
            if len(parts) >= 6 and parts[1] == '4':
                times.append(float(parts[0]))
                prices.append(int(parts[4]) / 10000.0)  # Convert back to float price
    return np.array(times), np.array(prices)

def seconds_to_hhmm(x, pos):
    hours = int(x // 3600)
    minutes = int((x % 3600) // 60)
    return f"{hours:02}:{minutes:02}"

def get_color_from_filename(filename):
    lower = filename.lower()
    if "ouup" in lower:
        return "green"
    elif "oudn" in lower:
        return "red"
    elif "oulvl" in lower:
        return "blue"
    else:
        return "gray"

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_OU_group.py '<glob_pattern>'")
        print("Example: python plot_OU_group.py 'lobster_trend_up_*.csv'")
        sys.exit(1)
    pattern = sys.argv[1]
    file_list = sorted(glob.glob(pattern))
    if not file_list:
        print(f"No files found matching pattern: {pattern}")
        sys.exit(1)

    plt.figure(figsize=(12, 6))
    for filename in file_list:
        times, prices = read_lobster_csv(filename)
        label = os.path.basename(filename)
        color = get_color_from_filename(label)
        plt.plot(times, prices, label=label, alpha=0.7, color=color)

    plt.xlabel("Time (HH:MM)")
    plt.ylabel("Price")
    plt.title("LOBSTER Execution Prices (Group)")
    plt.grid(True)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(seconds_to_hhmm))
    # Set major ticks every 1800 seconds (30 minutes)
    all_times = np.concatenate([read_lobster_csv(f)[0] for f in file_list])
    min_time = int(np.min(all_times))
    max_time = int(np.max(all_times))
    start_boundary = (min_time // 1800) * 1800
    end_boundary = ((max_time // 1800) + 1) * 1800
    ax.set_xticks(np.arange(start_boundary, end_boundary + 1, 1800))
    plt.tight_layout()
    #plt.legend(fontsize='small', loc='upper left', ncol=2)
    plt.show()

if __name__ == "__main__":
    main()
