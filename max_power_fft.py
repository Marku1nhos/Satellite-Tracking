"""
Doppler Extraction Pipeline for SatNOGS/CAMRAS IQ Recordings
------------------------------------------------------------

This script loads raw complex IQ data from the CAMRAS SatNOGS archive 
(16-bit interleaved I/Q at 48 kHz), converts it into complex floating-point 
samples, generates a spectrogram for visual inspection, and estimates the 
instantaneous Doppler frequency of a satellite pass.

Pipeline:

1. Load raw IQ (signed 16-bit integers, interleaved I/Q).
2. Normalise samples to [-1, 1] for numerical stability.
3. Compute a spectrogram (FFT-based time–frequency plot) using scipy.signal.spectrogram.
4. Apply a sliding-window FFT Doppler estimator to track the dominant narrowband 
   frequency in each window.
5. Convert the baseband frequency estimate into real RF frequency by adding 
   the SDR tuning frequency (rx-freq).
6. Plot both the baseband Doppler track and the absolute RF Doppler track.

This script provides a minimal end-to-end demonstration of Doppler extraction 
from real satellite IQ data and serves as a starting point for more advanced:
- signal detection,
- peak-locking algorithms,
- maneuver detection,
- or classification pipelines.

Inputs:
    iq_file (str): Path to the .raw IQ file from CAMRAS.
    fs (float): Sample rate of the IQ data (Hz).
    f_rx (float): RF tuning frequency of the SDR during the observation (Hz).

Outputs:
    - Spectrogram plot of the first few seconds.
    - Baseband Doppler curve.
    - Absolute RF Doppler curve (f_rx + baseband_offset).

Dependencies:
    numpy, matplotlib, scipy.signal, your custom estimate_doppler() function.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from signal_processing import make_synthetic_iq, estimate_doppler

##############################################################
# 1. PARAMETERS YOU MUST SET
##############################################################

iq_file = "test_iq_10963519.raw"      # <-- Your downloaded IQ file
fs = 48_000               # sample rate from metadata
f_rx = 436_885_600             # RF tuning frequency (Hz)

# Doppler estimator parameters
WIN_SIZE = 4096
OVERLAP = 0.5
USE_INTERP = True


##############################################################
# 2. LOAD IQ FILE
##############################################################

print("Loading IQ file:", iq_file)
raw = np.fromfile("test_iq_10963519.raw", dtype=np.int16)
# Interleave: even = I, odd = Q
I = raw[0::2].astype(np.float32)
Q = raw[1::2].astype(np.float32)

# Normalize (optional, good for spectrogram)
iq = (I + 1j*Q) / 32768.0

print(f"Loaded {len(iq)} samples "
      f"({len(iq)/fs:.2f} seconds of data)")


##############################################################
# 3. QUICK SPECTROGRAM TO CONFIRM SIGNAL
##############################################################

print("Plotting spectrogram...")

# Use first N seconds (or entire file)
N_sec = 480
N_samples = min(int(N_sec * fs), len(iq))

f_spec, t_spec, Sxx = spectrogram(
    iq[:N_samples],
    fs=fs,
    window="hann",
    nperseg=4096,
    noverlap=2048,
    detrend=False,
    scaling="density",
    mode="magnitude",
    return_onesided=False
)

# Shift frequency axis and magnitude (compatible syntax)
Sxx = np.fft.fftshift(Sxx, axes=0)
f_spec = np.fft.fftshift(f_spec)

Sxx_dB = 20 * np.log10(Sxx + 1e-12)

plt.figure(figsize=(10, 6))
plt.pcolormesh(t_spec, f_spec, Sxx_dB,
               shading='auto', cmap='viridis')
plt.title("Spectrogram of IQ (first few seconds)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz) relative to RF center (baseband)")
plt.colorbar(label="Magnitude (dB)")
plt.tight_layout()
plt.show()


##############################################################
# 5. RUN DOPPLER TRACKER
##############################################################

print("Running Doppler estimator...")

times, f_baseband = estimate_doppler(
    iq,
    fs=fs,
    win_size=WIN_SIZE,
    overlap=OVERLAP,
    use_interp=USE_INTERP
)

print("Estimated", len(times), "frequency points.")


##############################################################
# 6. CONVERT BASEBAND → REAL RF FREQUENCY
##############################################################

f_rf = f_rx + f_baseband   # real RF Doppler-shifted frequency


##############################################################
# 7. PLOT DOPPLER CURVE
##############################################################

plt.figure(figsize=(10, 5))
plt.plot(times, f_rf/1e6, 'o-', markersize=3)
plt.title("Real RF Doppler Curve")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (MHz)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(times, f_baseband, 'o-', markersize=3)
plt.title("Baseband Doppler Curve (relative to center)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency Offset (Hz)")
plt.grid(True)
plt.tight_layout()
plt.show()
