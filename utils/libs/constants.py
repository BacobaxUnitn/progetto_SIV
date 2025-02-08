import numpy as np
import pyaudio

chord_templates = {
    "C Major":   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # C-E-G
    "C# Major":  [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # C#-F-G#
    "D Major":   [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # D-F#-A
    "D# Major":  [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # D#-G-A#
    "E Major":   [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # E-G#-B
    "F Major":   [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # F-A-C
    "F# Major":  [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # F#-A#-C#
    "G Major":   [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # G-B-D
    "G# Major":  [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  # G#-C-D#
    "A Major":   [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # A-C#-E
    "A# Major":  [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # A#-D-F
    "B Major":   [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],  # B-D#-F#

    "C Minor":   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # C-E♭-G
    "C# Minor":  [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # C#-E-G#
    "D Minor":   [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # D-F-A
    "D# Minor":  [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # D#-F#-A#
    "E Minor":   [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # E-G-B
    "F Minor":   [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # F-G#-C
    "F# Minor":  [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # F#-A-C#
    "G Minor":   [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # G-A#-D
    "G# Minor":  [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  # G#-B-D#
    "A Minor":   [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # A-C-E
    "A# Minor":  [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # A#-C#-F
    "B Minor":   [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],  # B-D-F#
}


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0  # Return 0 instead of NaN if one vector is zero

    return np.dot(a, b) / (norm_a * norm_b)


CHUNK = 1024            # Number of audio samples per buffer
FORMAT = pyaudio.paInt16  # each sample is 16 bit
CHANNELS = 1            # Mono audio
RATE = 44100            # Sampling rate in Hz -> how many samples per second
RECORD_SECONDS = 5      # Duration of recording in seconds
AUDIO_FILE = './utils/assets/lewis_capaldi.mp3'
AUDIO_FILE_1 = './utils/assets/test_SIV.mp3'
NUMBER_OF_CHUNKS = int(RATE / CHUNK * RECORD_SECONDS)
WINDOW_SIZE = 2048
HOP_SIZE = 512