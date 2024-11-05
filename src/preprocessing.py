import numpy as np
from biosppy.signals import tools as tools

def filter_ecg(signal, sampling_rate):
    """
    Applies a bandpass filter to ECG signals.
    Parameters:
        signal (array-like): The raw ECG signal data.
        sampling_rate (int): The sampling rate of the ECG signal in Hz.
    Returns:
        filtered (ndarray): The bandpass-filtered ECG signal.
    """
    # Convert signal to a NumPy array for compatibility
    signal = np.array(signal)
    
    # Define the order of the FIR filter as a fraction of the sampling rate
    order = int(0.3 * sampling_rate)
    
    # Apply FIR bandpass filter (3-45 Hz) to isolate ECG frequency band
    filtered, _, _ = tools.filter_signal(
        signal=signal,
        ftype='FIR',
        band='bandpass',
        order=order,
        frequency=[3, 45],
        sampling_rate=sampling_rate
    )
    return filtered

def filter_ppg(signal, sampling_rate):
    """
    Applies a bandpass filter to PPG signals.
    Parameters:
        signal (array-like): The raw PPG signal data.
        sampling_rate (int or float): The sampling rate of the PPG signal in Hz.
    Returns:
        filtered (ndarray): The bandpass-filtered PPG signal.
    """
    # Convert signal to a NumPy array for compatibility
    signal = np.array(signal)
    
    # Ensure sampling_rate is a float for compatibility with filtering functions
    sampling_rate = float(sampling_rate)
    
    # Apply Butterworth bandpass filter (1-8 Hz) for PPG signal processing
    filtered, _, _ = tools.filter_signal(
        signal=signal,
        ftype='butter',
        band='bandpass',
        order=4,
        frequency=[1, 8],
        sampling_rate=sampling_rate
    )
    return filtered
