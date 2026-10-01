"""
Median Filter — Noise Reduction for Grid Sensor Readings
===========================================================
Chosen after testing 5 filtering methods against simulated sensor noise
(see noise_experiment.py). The median filter recovered the most accuracy
because it is naturally robust to outlier spikes (sensor glitches) —
unlike mean-based filters, which get pulled off by them.

How to use this in the pipeline:
  Instead of feeding a single raw sensor reading straight into the model,
  collect a short WINDOW of recent readings for each feature and pass the
  window through median_filter() first. Feed the filtered/cleaned values
  into the trained classifier (pipeline.py), not the raw ones.
"""

import numpy as np


def median_filter(readings: np.ndarray) -> np.ndarray:
    """
    Apply a median filter across a window of noisy sensor readings.

    Parameters
    ----------
    readings : np.ndarray, shape (n_samples, window_size, n_features)
        A short history of recent noisy readings for each sample and
        each of the 12 grid features (tau1-4, p1-4, g1-4).
        Example: 10 readings taken over the last 10 seconds.

    Returns
    -------
    np.ndarray, shape (n_samples, n_features)
        One cleaned estimate per feature, per sample — ready to feed
        into the trained classifier.
    """
    return np.median(readings, axis=1)


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    FEATURES = ["tau1", "tau2", "tau3", "tau4",
                "p1", "p2", "p3", "p4",
                "g1", "g2", "g3", "g4"]

    # --- Simulate one grid state being read 10 times in a row, noisily ---
    rng = np.random.default_rng(0)

    true_state = np.array([[2.5, 3.1, 4.0, 2.8,       # tau1-4
                             3.5, -1.2, -1.1, -1.2,    # p1-4
                             0.4, 0.6, 0.3, 0.5]])     # g1-4

    window_size = 10
    n_features = true_state.shape[1]

    # Gaussian noise on every reading
    noise = rng.normal(0, 0.15, size=(1, window_size, n_features))
    noisy_readings = true_state[:, None, :] + noise

    # Inject one outlier spike into reading #4 (simulating a sensor glitch)
    noisy_readings[0, 4, :] += rng.normal(0, 1.5, size=n_features)

    # --- Apply the filter ---
    cleaned = median_filter(noisy_readings)

    print("True state:       ", np.round(true_state[0], 3))
    print("Raw reading #1:    ", np.round(noisy_readings[0, 0], 3), " <- what 'no filter' would use")
    print("Reading with spike:", np.round(noisy_readings[0, 4], 3), " <- the glitch")
    print("Median filtered:   ", np.round(cleaned[0], 3), " <- what actually gets fed to the model")

    error_raw = np.abs(noisy_readings[0, 0] - true_state[0]).mean()
    error_filtered = np.abs(cleaned[0] - true_state[0]).mean()
    print(f"\nMean abs error, raw single reading: {error_raw:.4f}")
    print(f"Mean abs error, median filtered:     {error_filtered:.4f}")
