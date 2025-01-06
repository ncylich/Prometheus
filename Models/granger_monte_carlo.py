import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

import sys
if 'google.colab' in sys.modules:
    from Prometheus.Models.granger_causality import granger_causality
else:
    from granger_causality import granger_causality



NUM_SAMPLES = 1000
std = 0.002808538978948627
LENGTH = 17699
OBSERVED_P = 0.00004336
OBSERVED_F = 25.32


def random_test(*args, **kwargs):
    stocks = {i: np.random.normal(0, std, LENGTH) for i in range(8)}
    stocks = pd.DataFrame(stocks)

    min_p, max_f = 1, 0
    for i in range(8):
        for j in range(8):
            if i != j:
                p, f = granger_causality(stocks, i, j)
                min_p = min(min_p, p)
                max_f = max(max_f, f)
    return min_p, max_f


def main():
    p_values = []
    f_stats = []

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(random_test, range(NUM_SAMPLES)), total=NUM_SAMPLES))

    for p, f in results:
        p_values.append(p)
        f_stats.append(f)

    # Calculate percentiles for observed p-value and f-statistic
    p_values = np.array(p_values)
    f_stats = np.array(f_stats)
    p_percentile = np.mean(p_values <= OBSERVED_P)
    f_percentile = np.mean(f_stats >= OBSERVED_F)
    print(f'Observed P-value Percentile: {p_percentile * 100:.2f}%')
    print(f'Observed F-Statistic Percentile: {f_percentile * 100:.2f}%')

    # graph p-value distribution
    # take log of p_values for better visualization
    epsilon = 1e-10
    p_values = np.log(p_values + epsilon)
    observed_p = np.log(OBSERVED_P + epsilon)

    plt.hist(p_values, bins=20)
    plt.title('Ln of P-Value Distribution')
    plt.axvline(x=observed_p, color='r', linestyle='--')  # draw vertical line at observed p-value
    plt.show()

    # graph f-statistic distribution
    plt.hist(f_stats, bins=20)
    plt.title('F-Statistic Distribution')
    plt.axvline(x=OBSERVED_F, color='r', linestyle='--')  # draw a vertical line at observed f-statistic
    plt.show()

if __name__ == '__main__':
    main()
