import numpy as np

def make_synthetic_iq(fs, f0, drift_rate, duration, snr_db=0):
    t = np.arange(int(fs * duration)) / fs
    f_inst = f0 + drift_rate * t
    phase = 2 * np.pi * np.cumsum(f_inst) / fs
    iq = np.exp(1j * phase)
    # Add noise
    signal_power = np.mean(np.abs(iq)**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = (np.sqrt(noise_power/2) *
             (np.random.randn(*iq.shape) + 1j*np.random.randn(*iq.shape)))
    return t, f_inst, iq + noise

def estimate_doppler(iq, fs, win_size=4096, overlap=0.5, use_interp=True):
    step = int(win_size * (1 - overlap))
    n_windows = (len(iq) - win_size) // step

    window = np.hanning(win_size)
    times = []
    freqs_est = []

    for i in range(n_windows):
        start = i * step
        segment = iq[start:start + win_size] * window

        X = np.fft.fftshift(np.fft.fft(segment))
        freqs = np.fft.fftshift(np.fft.fftfreq(win_size, d=1/fs))
        mag = np.abs(X)
        k_max = np.argmax(mag)

        center_time = (start + win_size / 2) / fs

        if use_interp and 0 < k_max < len(mag) - 1:
            alpha = mag[k_max - 1]
            beta  = mag[k_max]
            gamma = mag[k_max + 1]
            denom = (alpha - 2*beta + gamma)
            if denom != 0:
                delta = 0.5 * (alpha - gamma) / denom
            else:
                delta = 0.0
        else:
            delta = 0.0

        df = fs / win_size
        f_peak = freqs[k_max] + delta * df

        times.append(center_time)
        freqs_est.append(f_peak)

    return np.array(times), np.array(freqs_est)