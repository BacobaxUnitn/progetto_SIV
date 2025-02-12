from scipy.ndimage import median_filter, gaussian_filter

from librosa.core.spectrum import stft
from librosa.core.pitch import estimate_tuning
from librosa import filters, util
import librosa

from pydub import AudioSegment

import numpy as np

from utils.libs.constants import cosine_similarity
from utils.libs.chords_score import ChordScores


def from_mp3(file_path, sr=22050):
    """
    Loads an MP3 file and converts it into a normalized waveform.

    Parameters:
        file_path (str): Path to the MP3 file.
        sr (int, optional): Sampling rate. Default is 22050 Hz.

    Returns:
        np.ndarray: Normalized waveform.
    """
    audio = AudioSegment.from_mp3(file_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(sr)
    raw_audio_data = np.frombuffer(audio.raw_data, dtype=np.int16)
    return raw_audio_data.astype(np.float32) / np.iinfo(np.int16).max  # Normalize to [-1,1]


def chroma_stft(*, y=None, sr=22050, norm=np.inf, n_fft=2048, hop_length=512, win_length=None, window="hann",
                center=True, pad_mode="constant", tuning=None, n_chroma=12, **kwargs):
    """
    Computes the chroma feature representation of an audio signal.

    Parameters:
        y (np.ndarray): Audio waveform.
        sr (int): Sampling rate.
        norm (float): Normalization factor.
        n_fft (int): FFT window size.
        hop_length (int): Hop length between frames.
        win_length (int, optional): Window length for STFT.
        window (str): Window function type.
        center (bool): Whether to center the signal.
        pad_mode (str): Padding mode.
        tuning (float, optional): Tuning offset.
        n_chroma (int): Number of chroma bins.

    Returns:
        np.ndarray: Normalized chroma representation.
    """
    S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center, window=window,
                    pad_mode=pad_mode)) ** 2

    if tuning is None:
        tuning = estimate_tuning(S=S, sr=sr, bins_per_octave=n_chroma)

    chromafb = filters.chroma(sr=sr, n_fft=n_fft, tuning=tuning, n_chroma=n_chroma, **kwargs)
    raw_chroma = np.einsum("cf,...ft->...ct", chromafb, S, optimize=True)
    return util.normalize(raw_chroma, norm=norm, axis=-2)


def smooth_chroma_median(chroma_matrix, kernel_size=3):
    """
    Applies median filtering to smooth the chroma matrix.
    """
    return median_filter(chroma_matrix, size=(1, kernel_size))


def smooth_chroma_gaussian(chroma_matrix, sigma=1):
    """
    Applies Gaussian filtering to smooth the chroma matrix.
    """
    return gaussian_filter(chroma_matrix, sigma=(0, sigma))


def tempo_and_duration(signal, sr):
    """
    Computes tempo and beat times from an audio signal.
    """
    tempo, beats = librosa.beat.beat_track(y=signal, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    beat_duration = 60 / tempo
    return tempo, beat_times, beat_duration


class ChromaStudio:
    """
    A class for managing chroma feature extraction and smoothing.
    """

    def __init__(self, file_path, kernel_size=3, sigma=1, sr=22050, hop_length=512, fft_size=2048):
        self.chroma, self.median_smoothed_chroma, self.gaussian_smoothed_chroma, self.y = create_and_smooth_chroma(
            file_path=file_path, kernel_size=kernel_size, sigma=sigma, sr=sr, hop_length=hop_length, fft_size=fft_size
        )
        self.sr = sr
        self.hop_length = hop_length
        self.fft_size = fft_size
        self.tempo, self.beat_times, self.beat_duration = tempo_and_duration(self.y, self.sr)
        self.chordScores = None

    def create_chord_scores(self, model="base", kernel=cosine_similarity):
        """
        Computes chord similarity scores using the chosen model.
        """
        self.chordScores = ChordScores(
            self.median_smoothed_chroma, self.index_to_time, self.get_period(), self.get_beat_duration(), kernel=kernel,
            model=model
        )
        return self

    def time_to_index(self, time):
        """
        Converts a given time in seconds to an index.
        """
        return int(time / (self.hop_length / self.sr))

    def index_to_time(self, index):
        """
        Converts an index to time in seconds.
        """
        return index * (self.hop_length / self.sr)

    def get_period(self):
        """
        Returns the time duration of one frame.
        """
        return self.hop_length / self.sr

    def get_tempo(self):
        """
        Returns the estimated tempo.
        """
        return self.tempo

    def get_beat_duration(self):
        """
        Returns the estimated beat duration.
        """
        return self.beat_duration

    def get_chord_scores(self) -> ChordScores:
        """
        Returns the computed chord scores object.
        """
        if self.chordScores is None:
            raise ValueError("You need to create the chord scores first.")
        return self.chordScores

    def plot_chord_heatmap(self, window: (int, int) = None):
        """
        Plots a heatmap of the chord progression over time.
        """
        self.get_chord_scores().plot_chord_heatmap(window)
