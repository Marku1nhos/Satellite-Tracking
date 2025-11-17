import numpy as np
import matplotlib.pyplot as plt

# 1. Generate a drifting complex tone (your synthetic Doppler signal)
fs = 1_000_000       # sample rate (1 MHz)
f0 = 100_000         # initial tone frequency (100 kHz)
drift_rate = 10_000  # Hz per second, big enough to see
duration = 0.05      # seconds

t = np.arange(int(fs * duration)) / fs
f_inst = f0 + drift_rate * t  # "true" instantaneous frequency vs time

# integrate frequency to get phase
phase = 2 * np.pi * np.cumsum(f_inst) / fs
iq = np.exp(1j * phase)

# 2. Sliding FFT windows to estimate frequency
win_size = 4096                  # samples per window
overlap = win_size // 2          # 50% overlap
step = win_size - overlap
n_windows = (len(iq) - win_size) // step

times = []
f_est = []

window = np.hanning(win_size)    # window function

for i in range(n_windows):
    start = i * step
    segment = iq[start:start + win_size] * window

    # FFT and shift so 0 Hz is in the middle
    # fftshift/fftfreq map bin indices to actual frequencies
    X = np.fft.fftshift(np.fft.fft(segment))
    freqs = np.fft.fftshift(np.fft.fftfreq(win_size, d=1/fs))

    mag = np.abs(X)
    k_max = np.argmax(mag)                     # k_max = index of the frequency bin with the highest powerv
    times.append((start + win_size / 2) / fs)  # time at window centre
    f_est.append(freqs[k_max])                 # peak frequency for this window

times = np.array(times)
f_est = np.array(f_est)

# 3. Plot "true" vs estimated frequency
plt.figure(figsize=(10, 6))
plt.plot(t, f_inst, label="True f(t)", linewidth=2)
plt.plot(times, f_est, 'o-', label="Estimated f(t) from FFT windows", markersize=4)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Doppler-like Frequency Drift: True vs Estimated")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
