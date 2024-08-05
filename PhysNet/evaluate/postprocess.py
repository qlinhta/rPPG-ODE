import numpy as np
import scipy
import scipy.io
from scipy.signal import butter, periodogram, find_peaks
from scipy.sparse import spdiags


def mag2db(mag):
    return 20. * np.log10(mag)


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def detrend(signal, lambda_value):
    """
    :param signal: T, or B x T
    :param lambda_value:
    :return:
    """
    T = signal.shape[-1]
    H = np.identity(T)
    ones = np.ones(T)
    minus_twos = -2 * np.ones(T)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (T - 2), T).toarray()
    designal = (H - np.linalg.inv(H + (lambda_value ** 2) * D.T.dot(D))).dot(signal.T).T
    return designal


def calculate_SNR(psd, freq, gtHR, target):
    """
    Signal-to-noise ratio
    :param psd: predict PSD
    :param freq: predict frequency
    :param gtHR: ground truth
    :param target: signal type
    """
    gtHR = gtHR / 60
    gtmask1 = (freq >= gtHR - 0.1) & (freq <= gtHR + 0.1)
    gtmask2 = (freq >= gtHR * 2 - 0.1) & (freq <= gtHR * 2 + 0.1)
    sPower = psd[np.where(gtmask1 | gtmask2)].sum()
    if target == 'pulse':
        mask = (freq >= 0.75) & (freq <= 4)
    else:
        mask = (freq >= 0.08) & (freq <= 0.5)
    allPower = psd[np.where(mask)].sum()
    ret = mag2db(sPower / (allPower - sPower))
    return ret


def fft_physiology(signal: np.ndarray, target="pulse", Fs=30, diff=True, detrend_flag=True):
    """
    Calculate HR or FR using fft
    :param signal: T, or B x T
    :param target: pulse or respiration
    :param Fs:
    :param diff: differential signal
    :param detrend_flag: need for detrending
    :return:
    """
    if diff:
        signal = signal.cumsum(axis=-1)
    if detrend_flag:
        signal = detrend(signal, 100)
    if target == "pulse":
        [b, a] = butter(1, [0.75 / Fs * 2, 2.5 / Fs * 2], btype='bandpass')
    else:
        [b, a] = butter(1, [0.08 / Fs * 2, 0.5 / Fs * 2], btype='bandpass')
    signal = scipy.signal.filtfilt(b, a, np.double(signal))
    N = next_power_of_2(signal.shape[-1])
    freq, psd = periodogram(signal, fs=Fs, nfft=N, detrend=False)
    if target == "pulse":
        mask = np.argwhere((freq >= 0.75) & (freq <= 2.5))
    else:
        mask = np.argwhere((freq >= 0.08) & (freq <= 0.5))
    freq = freq[mask]
    if len(signal.shape) == 1:
        idx = psd[mask.reshape(-1)].argmax(-1)
    else:
        idx = psd[:, mask.reshape(-1)].argmax(-1)
    phys = freq[idx] * 60
    return phys.reshape(-1)


def peak_physiology(signal: np.ndarray, target="pulse", Fs=30, diff=True, detrend_flag=True):
    """
    Calculate HR or FR using ibi
    :param signal: T, or B x T
    :param target: pulse or respiration
    :param Fs:
    :param diff: differential signal
    :param detrend_flag: need for detrending
    :return:
    """
    if diff:
        signal = signal.cumsum(axis=-1)
    if detrend_flag:
        signal = detrend(signal, 100)
    if target == 'pulse':
        [b, a] = butter(1, [0.75 / Fs * 2, 2.5 / Fs * 2], btype='bandpass')
    else:
        [b, a] = butter(1, [0.08 / Fs * 2, 0.5 / Fs * 2], btype='bandpass')
    signal = scipy.signal.filtfilt(b, a, np.double(signal))
    T = signal.shape[-1]
    signal = signal.reshape(-1, T)
    phys = []
    for s in signal:
        peaks, _ = find_peaks(s)
        phys.append(60 * Fs / np.diff(peaks).mean(axis=-1))
    return np.asarray(phys)
