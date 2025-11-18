import numpy as np
import matplotlib.pyplot as plt

# 1. Generate a drifting complex tone (your synthetic Doppler signal)
fs = 1_000_000       # sample rate (1 MHz)
f0 = 100_000         # initial tone frequency (100 kHz)
drift_rate = 10_000  # Hz per second, big enough to see
duration = 0.2       # seconds

t = np.arange(int(fs * duration)) / fs
f_inst = f0 + drift_rate * t  # "true" instantaneous frequency vs time

# IQ signal
phase = 2 * np.pi * np.cumsum(f_inst) / fs
iq = np.exp(1j * phase)

# Window sizes to compare
win_sizes = [1024, 4096, 16384]

fig, axes = plt.subplots(len(win_sizes), 1,
                         figsize=(10, 4 * len(win_sizes)),
                         sharex=True, sharey=True)

if len(win_sizes) == 1:
    axes = [axes]

for ax, win_size in zip(axes, win_sizes):
    overlap = win_size // 2          # 50% overlap
    step = win_size - overlap
    # number of windows (include last full window)
    n_windows = max(1, (len(iq) - win_size) // step + 1)

    times = []
    f_est = []

    window = np.hanning(win_size)

    for i in range(n_windows):
        start = i * step
        segment = iq[start:start + win_size] * window
        X = np.fft.fftshift(np.fft.fft(segment))
        freqs = np.fft.fftshift(np.fft.fftfreq(win_size, d=1 / fs))
        mag = np.abs(X)
        k_max = np.argmax(mag)
        times.append((start + win_size / 2) / fs)
        f_est.append(freqs[k_max])

    times = np.array(times)
    f_est = np.array(f_est)

    # Plot
    ax.plot(t, f_inst, label="True f(t)", linewidth=1.5)
    ax.plot(times, f_est, 'o-', label="Estimated f(t) from FFT windows", markersize=4)
    df = fs / win_size
    dt = step / fs
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"win_size={win_size}, df≈{df:.1f} Hz, time step≈{dt*1e3:.2f} ms")
    ax.grid(True)
    ax.legend(loc="upper left")

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.show()
