import numpy as np

def estimate_doppler(iq, fs, win_size=4096, overlap=0.5, use_interp=True):
    """
    Estimate the Doppler-shifted instantaneous frequency of a signal over time.

    This function takes a complex baseband IQ signal (as recorded from an SDR or 
    generated synthetically) and measures how the dominant frequency changes over time.
    It does this by sliding a window across the signal, performing an FFT on each 
    window, finding the strongest frequency component in that window, and recording 
    both the time of the window and the estimated frequency.

    Parameters
    ----------
    iq : numpy array of complex
        The complex IQ samples of the signal. Each element is one complex sample 
        representing the signal at a single time step.

    fs : float
        Sampling rate of the IQ data in Hz (samples per second).

    win_size : int, optional
        Number of samples in each FFT window. Larger values give finer frequency 
        resolution but worse time resolution. Smaller values give better time 
        resolution but coarser frequency resolution. Default is 4096.

    overlap : float in [0, 1), optional
        Fraction of each window that overlaps with the next window. For example, 
        overlap=0.5 means each window starts halfway inside the previous one. 
        Overlap increases the number of Doppler estimates without sacrificing 
        window length. Default is 0.5.

    use_interp : bool, optional
        If True, perform sub-bin interpolation (a small parabolic fit around the 
        largest FFT bin) to estimate the peak frequency more accurately than the 
        raw FFT bin spacing. This greatly improves frequency precision. Default True.

    Returns
    -------
    times : numpy array (float)
        Array of timestamps (in seconds) corresponding to the center of each window.

    freqs_est : numpy array (float)
        Estimated frequency (in Hz) for each window. These form a tracked 
        Doppler-vs-time curve.

    Notes
    -----
    - The FFT converts each time window into a frequency spectrum.
    - The strongest frequency bin is assumed to be the signal's instantaneous frequency.
    - Because FFT bins are discrete, interpolation can estimate frequencies between bins.
    - This function is the core of a simple Doppler tracker: it converts an IQ 
      recording into a time series of frequency estimates.
    """
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
