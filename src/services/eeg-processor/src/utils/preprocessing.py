import numpy as np

def normalize_eeg_data(eeg_data):
    """Normalize EEG data to have zero mean and unit variance."""
    mean = np.mean(eeg_data, axis=0)
    std = np.std(eeg_data, axis=0)
    normalized_data = (eeg_data - mean) / std
    return normalized_data

def filter_eeg_data(eeg_data, low_cutoff, high_cutoff, sampling_rate):
    """Apply a bandpass filter to the EEG data."""
    from scipy.signal import butter, filtfilt

    nyquist = 0.5 * sampling_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(1, [low, high], btype='band')
    filtered_data = filtfilt(b, a, eeg_data, axis=0)
    return filtered_data

def segment_eeg_data(eeg_data, segment_length):
    """Segment EEG data into smaller chunks."""
    n_segments = len(eeg_data) // segment_length
    segments = np.array_split(eeg_data[:n_segments * segment_length], n_segments)
    return segments

def extract_features(eeg_segments):
    """Extract features from EEG segments."""
    features = []
    for segment in eeg_segments:
        mean = np.mean(segment, axis=0)
        std = np.std(segment, axis=0)
        features.append(np.concatenate((mean, std)))
    return np.array(features)