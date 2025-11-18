import numpy as np
import matplotlib.pyplot as plt
from signal_processing import make_synthetic_iq, estimate_doppler

if __name__ == "__main__":
    # Parameters (match crude_doppler.py)
    fs = 1_000_000
    f0 = 100_000
    drift_rate = 10_000
    duration = 0.05
    snr_db = 0

    t, f_inst, iq = make_synthetic_iq(fs, f0, drift_rate, duration, snr_db)

    win_sizes = [1024, 4096, 16384]
    overlap = 0.5

    fig, axes = plt.subplots(len(win_sizes), 1,
                             figsize=(10, 4 * len(win_sizes)),
                             sharex=True, sharey=True)
    if len(win_sizes) == 1:
        axes = [axes]

    for ax, win_size in zip(axes, win_sizes):
        times_interp, freq_interp = estimate_doppler(iq, fs, win_size=win_size,
                                                     overlap=overlap, use_interp=True)
        times_nointerp, freq_nointerp = estimate_doppler(iq, fs, win_size=win_size,
                                                         overlap=overlap, use_interp=False)

        ax.plot(t, f_inst, label="True f(t)", color="k", linewidth=1.2)
        ax.plot(times_nointerp, freq_nointerp, 'o-', label="Estimated (no interp)", markersize=4, alpha=0.8)
        ax.plot(times_interp, freq_interp, 's--', label="Estimated (parabolic interp)", markersize=4, alpha=0.9)

        df = fs / win_size
        step = int(win_size * (1 - overlap))
        dt = step / fs

        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"win_size={win_size}, df≈{df:.1f} Hz, time step≈{dt*1e3:.2f} ms")
        ax.grid(True)
        ax.legend(loc="upper left")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()