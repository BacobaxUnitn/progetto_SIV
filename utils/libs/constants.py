import numpy as np
import pyaudio

# Dictionary mapping chord names to their respective pitch-class templates
chord_templates = {
    "C Major": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # C-E-G
    "C# Major": [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # C#-F-G#
    "D Major": [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # D-F#-A
    "D# Major": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # D#-G-A#
    "E Major": [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # E-G#-B
    "F Major": [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # F-A-C
    "G Major": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # G-B-D
    "A Major": [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # A-C#-E
    "B Major": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],  # B-D#-F#

    "C Minor": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # C-E♭-G
    "D Minor": [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # D-F-A
    "E Minor": [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # E-G-B
    "F Minor": [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # F-G#-C
    "G Minor": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # G-A#-D
    "A Minor": [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # A-C-E
    "B Minor": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],  # B-D-F#
}


def cosine_similarity(a, b):
    """
    Computes the cosine similarity between two vectors.

    Parameters:
        a (array-like): First vector.
        b (array-like): Second vector.

    Returns:
        float: Cosine similarity score.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0  # Return 0 instead of NaN if one vector is zero
    return np.dot(a, b) / (norm_a * norm_b)


def weighted_jaccard(vec, template):
    """
    Computes the weighted Jaccard similarity between a frequency vector and a chord template.

    Parameters:
        vec (array-like): Input frequency magnitude vector.
        template (array-like): Chord template vector.

    Returns:
        float: Weighted Jaccard similarity score.
    """
    vec = np.array(vec)
    template = np.array(template)
    intersection = np.sum(np.minimum(vec, template))
    union = np.sum(np.maximum(vec, template))
    return intersection / union if union > 0 else 0


def similarity_scores(freq_array, kernel=cosine_similarity):
    """
    Computes similarity scores between a given frequency vector and all chord templates.

    Parameters:
        freq_array (array-like): Frequency magnitude vector of the input.
        kernel (function, optional): Similarity function to use. Default is cosine_similarity.

    Returns:
        dict: Dictionary with chord names as keys and similarity scores as values.
    """
    scores = {}
    for chord_name, chord_template in chord_templates.items():
        scores[chord_name] = kernel(freq_array, chord_template)
    return scores


# Audio processing constants
CHUNK = 1024  # Number of audio samples per buffer
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate in Hz
RECORD_SECONDS = 5  # Duration of recording in seconds
AUDIO_FILE = './utils/assets/lewis_capaldi.mp3'
AUDIO_FILE_1 = './utils/assets/test_SIV.mp3'
AUDIO_FILE_2 = './utils/assets/nuv_bianche.mp3'
NUMBER_OF_CHUNKS = int(RATE / CHUNK * RECORD_SECONDS)
WINDOW_SIZE = 2048
HOP_SIZE = 512