from scipy.ndimage import median_filter, gaussian_filter

from librosa.core.spectrum import stft
from librosa.core.pitch import estimate_tuning
from librosa import filters
from librosa import util
import librosa

from pydub import AudioSegment

import numpy as np

from utils.libs.conversions import get_index_time_mapping


def from_mp3(file_path, sr=22050):
    audio = AudioSegment.from_mp3(file_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(sr)
    raw_audio_data = np.frombuffer(audio.raw_data, dtype=np.int16)
    return raw_audio_data.astype(np.float32) / np.iinfo(np.int16).max  # Normalize to [-1,1]


def chroma_stft(*, y=None, sr: float = 22050, S=None, norm=np.inf, n_fft=2048, hop_length=512, win_length=None, window="hann", center: bool = True, pad_mode="constant", tuning=None, n_chroma: int = 12, **kwargs,):
    power = 2
    S = (
            np.abs(
                stft(
                    y,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    center=center,
                    window=window,
                    pad_mode=pad_mode,
                )
            )
            ** power
    )

    if tuning is None:
        tuning = estimate_tuning(S=S, sr=sr, bins_per_octave=n_chroma)

    # Get the filter bank
    chromafb = filters.chroma(
        sr=sr, n_fft=n_fft, tuning=tuning, n_chroma=n_chroma, **kwargs
    )

    # Compute raw chroma
    # EQUIVALENT TO raw_chroma = np.dot(chromafb, S)
    # (bins , frequencies) x (frequencies, time) -> (bins, time)
    raw_chroma = np.einsum("cf,...ft->...ct", chromafb, S, optimize=True)

    return util.normalize(raw_chroma, norm=norm, axis=-2)


def smooth_chroma_median(chroma_matrix, kernel_size=3):
    return median_filter(chroma_matrix, size=(1, kernel_size))  # Apply along time axis


def smooth_chroma_gaussian(chroma_matrix, sigma=1):
    return gaussian_filter(chroma_matrix, sigma=(0, sigma))  # Apply along time axis


def create_and_smooth_chroma(file_path, kernel_size=3, sigma=1, RATE=22050):
    audio = from_mp3(file_path)
    chroma = chroma_stft(y=audio, sr=RATE)
    median_smoothed_chroma = smooth_chroma_median(chroma, kernel_size)
    gaussian_smoothed_chroma = smooth_chroma_gaussian(chroma, sigma)
    return chroma, median_smoothed_chroma, gaussian_smoothed_chroma, audio


def create_chroma_studio(file_path, kernel_size=3, sigma=1, sr=22050, hop_length=512, fft_size=2048):
    chroma, median_smoothed_chroma, gaussian_smoothed_chroma, y = create_and_smooth_chroma(
        file_path,
        kernel_size,
        sigma,
        sr
    )
    time_2_idx, idx_2_time, cut_chroma_time = get_index_time_mapping(512, sr)
    return ChromaStudio(
        chroma,
        median_smoothed_chroma,
        gaussian_smoothed_chroma,
        time_2_idx,
        idx_2_time,
        cut_chroma_time,
        y,
        sr,
        hop_length,
        fft_size,

    )


def tempo_and_duration(signal, sr):
    tempo, beats = librosa.beat.beat_track(y=signal, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    beat_duration = 60 / tempo  # Seconds per beat
    return tempo, beat_times, beat_duration


class ChromaStudio:
    def __init__(
            self,
            chroma,
            median_chroma,
            gaussian_chroma,
            time_2_idx,
            idx_2_time,
            cut_chroma_time,
            y,
            sr=22050,
            hop_length=512,
            fft_size=2048,

            ):
        self.chroma = chroma
        self.median_smoothed_chroma = median_chroma
        self.gaussian_smoothed_chroma = gaussian_chroma
        self.time_2_idx = time_2_idx
        self.idx_2_time = idx_2_time
        self.cut_chroma_time = cut_chroma_time
        self.sr = sr
        self.hop_length = hop_length
        self.fft_size = fft_size
        self.y = y
        self.tempo, self.beat_times, self.beat_duration = tempo_and_duration(y, sr)



    def time_to_index(self,time):
        """
        Converts time to index.

        Parameters:
        - time (float): Time in seconds.
        - sr (int): Sampling rate of the audio.
        - hop_length (int): Hop size used in STFT.

        Returns:
        - int: Index corresponding to the time.
        """
        time_per_column = self.hop_length / self.sr
        return int(time / time_per_column)

    def index_to_time(self, index):
        """
        Converts index to time.

        Parameters:
        - index (int): Index.
        - sr (int): Sampling rate of the audio.
        - hop_length (int): Hop size used in STFT.

        Returns:
        - float: Time in seconds corresponding to the index.
        """
        time_per_column = self.hop_length / self.sr
        return index * time_per_column

    def cut_time(self, chroma_matrix, start_time, end_time):
        """
        Cuts a time segment from a chroma matrix.

        Parameters:
        - chroma_matrix (np.ndarray): The chroma matrix (shape: 12 x num_time_frames).
        - start_time (float): The start time in seconds.
        - end_time (float): The end time in seconds.
        - sr (int): Sampling rate of the audio.
        - hop_length (int): Hop size used in STFT.

        Returns:
        - np.ndarray: Cropped chroma matrix.
        """

        # Convert times to indices
        start_index = self.time_to_index(start_time)
        end_index = self.time_to_index(end_time)

        # Ensure indices are within bounds
        start_index = max(0, start_index)
        end_index = min(chroma_matrix.shape[1], end_index)

        # Slice the chroma matrix
        return chroma_matrix[:, start_index:end_index]

    def get_chroma(self):
        return self.chroma

    def get_median_smoothed_chroma(self):
        return self.median_smoothed_chroma

    def get_gaussian_smoothed_chroma(self):
        return self.gaussian_smoothed_chroma

    def get_time_2_idx(self):
        return self.time_2_idx

    def get_idx_2_time(self):
        return self.idx_2_time

    def get_cut_chroma_time(self):
        return self.cut_chroma_time

    def get_sr(self):
        return self.sr

    def get_hop_length(self):
        return self.hop_length

    def get_fft_size(self):
        return self.fft_size

    def get_y(self):
        return self.y

    def get_period(self):
        return self.hop_length / self.sr

    def get_tempo(self):
        return self.tempo

    def get_beat_times(self):
        return self.beat_times

    def get_beat_duration(self):
        return self.beat_duration



