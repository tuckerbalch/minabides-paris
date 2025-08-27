# Copyright (c) 2025 Tucker Balch
#
# Generation of this code was assisted by GitHub Copilot version 1.300.0

"""
This script generates LOBSTER-style CSV files for use by the minABIDES simulator. It mostly 
generates type 4 buy events (execution events) with a few type 1 events (new orders) at the beginning.
The orders at the beginning simulate pre-market trading to populate the order book before the
market opens. The transaction prices are intended to be used as "the fundamental" by agents
participating in the simulated market. 

It simulates a mean-reverting process using the Ornstein-Uhlenbeck process and adds random 
shocks to the process. The first value is always 100.00 and that is also the long-term mean of
the OU process.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

MARKET_OPEN = 9 * 3600 + 30 * 60  # Market opens at 9:30 AM
MARKET_CLOSE = 16 * 3600  # Market closes at 4:00 PM
SECONDS_PER_TRADING_DAY = MARKET_CLOSE - MARKET_OPEN  # Number of seconds in a day

def gen_times(num_times, start_time, end_time, seed=None):
    """
    Generates an array of random times during the day according to a Poisson process.
    Important note: It may generate more times than requested, due to
    the upredictable nature of the Poisson process. The last time will be >= end_time.

    Parameters:
        num_times (int): Number of times to generate.
        start_time (float): The first time (in seconds since midnight).
        end_time (float): The last time (in seconds since midnight).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        numpy.ndarray: Array of random times (in seconds since midnight) sorted in ascending order.
    """
   
    if seed is not None:
        np.random.seed(seed)
    if start_time >= end_time:
        raise ValueError("start_time must be less than end_time")
    avg_rate = num_times / (end_time - start_time)  # Average rate of events per second
    inter_arrival_times = np.random.exponential(scale=1/avg_rate, size=num_times)
    inter_arrival_times[0] = 0  # Set the first inter-arrival time to 0 so that we start at start_time
    event_times = np.cumsum(inter_arrival_times)
    event_times = start_time + event_times

    # handle boundary conditions at the end of the list of times.
    while len(event_times) < num_times or event_times[-1] < end_time:
        additional_inter_arrival_times = np.random.exponential(scale=1/avg_rate, size = 10 ) # add 10 more
        additional_event_times = event_times[-1] + np.cumsum(additional_inter_arrival_times)
        event_times = np.concatenate((event_times, additional_event_times))

    return event_times

def compute_ou_process(times, mu, Q0, gamma, sigma, shock_rate, shock_mag, shock_std, trend=0.0, seed=None):
    """
    Computes a mean-reverting fundamental based on the Ornstein-Uhlenbeck (OU) Process,
    with shocks applied at times determined by a Poisson process. It may return more times
    than requested, due to additional shock times.

    Parameters:
        times (numpy.ndarray): Array of time points at which the process is evaluated.
        mu (float): Long-term mean value of the OU process.
        Q0 (float): Initial value of the OU process.
        gamma (float): Rate of mean reversion (speed at which the process reverts to the mean).
        sigma (float): Volatility parameter of the OU process.
        shock_rate (float): Rate parameter (lambda) of the Poisson process for generating shock times.
        shock_mag (float): Mean magnitude of shocks applied to the process.
        shock_std (float): Standard deviation of the shock magnitude.
        trend (float): Total amount to add to mu over the simulation (mu increases linearly from 0 to trend).
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple: Updated times array and OU process values.
    """
    if seed is not None:
        np.random.seed(seed)

    # Check if shock_rate is zero
    if shock_rate == 0:
        print("Warning: shock_rate is 0. No shocks will be applied.")
        shock_times = []
    else:
        # Generate Poisson shock times
        shock_times = []
        current_time = times[0]
        while current_time < times[-1]:
            inter_arrival_time = np.random.exponential(scale=1/shock_rate)
            current_time += inter_arrival_time
            if current_time < times[-1]:
                shock_times.append(current_time)

    # Combine shock times with the original times
    times = np.sort(np.concatenate((times, shock_times)))

    # Initialize the OU process
    Q = np.zeros(len(times))
    Q[0] = Q0  # Set the initial value

    # Calculate the incremental change in mu per step for the trend
    mu_trend = np.linspace(0, trend, len(times))

    # Iterate through the time steps
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]  # Time difference
        mu_i = mu + mu_trend[i]       # Adjust mu by the trend at this step
        mean = mu_i + (Q[i - 1] - mu_i) * np.exp(-gamma * dt)  # Mean reversion
        variance = (sigma**2 / (2 * gamma)) * (1 - np.exp(-2 * gamma * dt))  # Variance
        Q[i] = np.random.normal(loc=mean, scale=np.sqrt(variance))

        # Apply a shock if the current time matches a shock time
        if times[i] in shock_times:
            Q[i] += np.random.choice([-1, 1]) * np.random.normal(loc=shock_mag, scale=shock_std)
            print(f"Shock applied at time {times[i]:.2f}: {Q[i]:.2f}")

    return times, Q

def save_lobster(filename, times, values, event_type=4):
    """
    Saves the generated times and OU process values to a LOBSTER-style CSV file.
    Adds 10 buy orders of size 10 with prices ranging from 99.90 to 99.99
    and 10 sell orders of size 10 with prices ranging from 100.01 to 100.10
    at one second before the first time.

    Parameters:
        filename (str): Name of the output file.
        times (numpy.ndarray): Array of times (in seconds since midnight).
        values (numpy.ndarray): Array of OU process values.
        event_type (int, optional): Event type to save in the file (default is 4 for execution events).
    """
    with open(filename, 'w') as f:
        # Determine the time for the 10 buy and sell orders (one second before the first time in times)
        initial_time = max(0, times[0] - 1)  # Ensure the time is not negative

        # Add 10 buy orders at the beginning
        for i in range(10):
            price_in_cents = int((99.90 + i * 0.01) * 10000)  # Prices from 99.90 to 99.99
            f.write(f"{initial_time:.6f},1,0,100,{price_in_cents},1\n")  # Event type 1 (new order), size 100, direction 1 (buy)

        # Add 10 sell orders at the beginning
        for i in range(10):
            price_in_cents = int((100.01 + i * 0.01) * 10000)  # Prices from 100.01 to 100.10
            f.write(f"{initial_time:.6f},1,0,100,{price_in_cents},-1\n")  # Event type 1 (new order), size 100, direction -1 (sell)

        # Write the generated OU process data
        for i in range(len(times)):
            # Convert price to cents (LOBSTER format uses integer prices in cents)
            price_in_cents = int(values[i] * 10000)
            # Write the row in LOBSTER format: time, event_type, order_id, size, price, direction (buy)
            f.write(f"{times[i]:.6f},{event_type},0,0,{price_in_cents},1\n")

def main():
    """
    Main function to generate random times, compute the OU process, and save the result to a LOBSTER-style file.
    """
    parser = argparse.ArgumentParser(description="Generate LOBSTER-style CSV files using an OU process.")
    parser.add_argument("--num_times", type=int, default=5000, help="Number of times to generate.")
    parser.add_argument("--start_time", type=float, default=MARKET_OPEN, help="Start time in seconds since midnight.")
    parser.add_argument("--end_time", type=float, default=MARKET_CLOSE + 500, help="End time in seconds since midnight.")
    parser.add_argument("--seed", type=int, default=46, help="Random seed for reproducibility.")
    parser.add_argument("--mu", type=float, default=100, help="Long-term mean value of the OU process.")
    parser.add_argument("--Q0", type=float, default=100, help="Initial value of the OU process.")
    parser.add_argument("--gamma", type=float, default=0.0005, help="Mean reversion rate.")
    parser.add_argument("--sigma", type=float, default=0.005, help="Volatility of the OU process.")
    parser.add_argument("--shocks_per_day", type=float, default=3, help="Average number of shocks per day.")
    parser.add_argument("--shock_mag", type=float, default=0.05, help="Mean magnitude of shocks.")
    parser.add_argument("--shock_std", type=float, default=0.1, help="Standard deviation of shock magnitude.")
    parser.add_argument("--trend", type=float, default=0.0, help="Total increase in the fundamental over the simulation (e.g., 10 for a final value of 110 if mu=100).")
    parser.add_argument("--output_file", type=str, default="synthetic_lobster_data.csv", help="Output CSV filename.")

    args = parser.parse_args()

    shock_rate = args.shocks_per_day / SECONDS_PER_TRADING_DAY

    random_times = gen_times(args.num_times, args.start_time, args.end_time, seed=args.seed)
    print(f"Minimum time: {min(random_times):.2f} seconds ({int(divmod(min(random_times), 3600)[0]):02}:{int(divmod(min(random_times), 3600)[1] // 60):02} HH:MM), "
          f"Maximum time: {max(random_times):.2f} seconds ({int(divmod(max(random_times), 3600)[0]):02}:{int(divmod(max(random_times), 3600)[1] // 60):02} HH:MM)")

    updated_times, ou_values = compute_ou_process(
        random_times, args.mu, args.Q0, args.gamma, args.sigma,
        shock_rate, args.shock_mag, args.shock_std, trend=args.trend, seed=args.seed
    )

    save_lobster(args.output_file, updated_times, ou_values)

if __name__ == "__main__":
    main()

