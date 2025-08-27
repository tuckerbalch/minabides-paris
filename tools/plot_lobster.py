# Copyright (c) 2025 Tucker Balch
#
# Generation of this code was assisted by GitHub Copilot version 1.300.0

"""
This script plots a LOBSTER-style CSV file.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_lobster.py <csv_filename>")
        sys.exit(1)
    filename = sys.argv[1]
    times, prices = read_lobster_csv(filename)
    plt.figure(figsize=(12, 6))
    plt.plot(times, prices, color='blue', label='Execution Price')
    plt.xlabel("Time (HH:MM)")
    plt.ylabel("Price")
    plt.title("LOBSTER Execution Prices")
    plt.legend()
    plt.grid(True)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(seconds_to_hhmm))
    # Set major ticks every 1800 seconds (30 minutes)
    min_time = int(np.min(times))
    max_time = int(np.max(times))
    start_boundary = (min_time // 1800) * 1800
    end_boundary = ((max_time // 1800) + 1) * 1800
    ax.set_xticks(np.arange(start_boundary, end_boundary + 1, 1800))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()