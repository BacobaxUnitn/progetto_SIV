import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.libs.constants import cosine_similarity


def pitch_class_helix(k, r=1, h=0.5):
    """
    Computes the 3D position of a pitch class on the spiral array.
    """
    x = r * np.sin(k * np.pi / 2)
    y = r * np.cos(k * np.pi / 2)
    z = k * h
    return np.array([x, y, z])


def major_chord_helix(k, w1=0.5, w2=0.3, w3=0.2):
    """
    Computes the center of effect for a major chord (root, major third, perfect fifth).
    """
    return w1 * pitch_class_helix(k) + w2 * pitch_class_helix(k + 1) + w3 * pitch_class_helix(k + 4)


def minor_chord_helix(k, u1=0.5, u2=0.3, u3=0.2):
    """
    Computes the center of effect for a minor chord (root, minor third, perfect fifth).
    """
    return u1 * pitch_class_helix(k) + u2 * pitch_class_helix(k + 1) + u3 * pitch_class_helix(k + 9)


def synth_major_chord_helix_v2(frequencies, k_chord, w1=0.5, w2=0.3, w3=0.2):
    """
    Computes the center of effect for a major chord (root, major third, perfect fifth).
    """
    synth_pitch_freqs = np.array([synth_pitch_class_helix(k, m) for k, m in enumerate(frequencies)])
    return w1 * synth_pitch_freqs[k_chord] + w2 * synth_pitch_freqs[(k_chord + 1) % 12] + w3 * synth_pitch_freqs[(k_chord + 4) % 12]


def synth_major_chord_helix(frequencies, k_chord):
    """
    Computes the center of effect for a major chord (root, major third, perfect fifth).
    """
    w1 = frequencies[k_chord]
    w2 = frequencies[(k_chord + 1) % 12]
    w3 = frequencies[(k_chord + 4) % 12]

    return w1 * pitch_class_helix(k_chord) + w2 * pitch_class_helix(k_chord + 1) + w3 * pitch_class_helix(k_chord + 4)


def synth_minor_chord_helix_v2(frequencies, k_chord, u1=0.5, u2=0.3, u3=0.2):
    """
    Computes the center of effect for a minor chord (root, minor third, perfect fifth).
    """
    synth_pitch_freqs = np.array([synth_pitch_class_helix(k, m) for k, m in enumerate(frequencies)])
    return u1 * synth_pitch_freqs[k_chord] + u2 * synth_pitch_freqs[(k_chord + 1) % 12] + u3 * synth_pitch_freqs[(k_chord - 5) % 12]


def synth_minor_chord_helix(frequencies, k_chord):
    """
    Computes the center of effect for a minor chord (root, minor third, perfect fifth).
    """
    u1 = frequencies[k_chord]
    u2 = frequencies[(k_chord + 1) % 12]
    u3 = frequencies[(k_chord - 5) % 12]

    return u1 * pitch_class_helix(k_chord) + u2 * pitch_class_helix(k_chord + 1) + u3 * pitch_class_helix(k_chord - 5)


def synth_pitch_class_helix(k, w):
    tmp = pitch_class_helix(k)
    tmp[0] *= w
    tmp[1] *= w
    tmp[2] *= w
    return tmp


synth_chord_factory = {
    "major": synth_major_chord_helix,
    "minor": synth_minor_chord_helix
}

chord_factory = {
    "major": major_chord_helix,
    "minor": minor_chord_helix
}


def chroma_frequency_to_chord_similarity(frequencies, k_chord, chord_type, kernel=cosine_similarity):
    """
    :param chord_type:
    :param kernel: symmilarity measure between synthetic chord and non syntetic chord
    :param frequencies: array if size 12 with the chroma frequencies magnitudes
    :param k_chord: pitch of the major chord
    :return: a similarity measure between the chroma frequencies and the major chord
    """

    synth_chord = synth_chord_factory[chord_type](frequencies, k_chord)

    normal_chord = chord_factory[chord_type](k_chord)

    return kernel(normal_chord, synth_chord)


def sort_by_circle_of_fifths(pitch_magnitudes):
    """
    Sorts a 12-element array of pitch magnitudes from linear chromatic order
    into the order of the circle of fifths.

    :param pitch_magnitudes: List of 12 pitch magnitudes in chromatic order
                             [C, C#, D, D#, E, F, F#, G, G#, A, A#, B]
    :return: List of 12 pitch magnitudes sorted in circle of fifths order
             [C, G, D, A, E, B, F#, C#, G#, D#, A#, F]
    """

    # Circle of Fifths mapping indices
    circle_of_fifths_order = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]

    # Rearrange pitch magnitudes
    sorted_magnitudes = [pitch_magnitudes[i] for i in circle_of_fifths_order]

    return sorted_magnitudes


def sort_by_chromatic_order(pitch_magnitudes):
    """
    Reverses a 12-element array of pitch magnitudes from the order of the
    circle of fifths back to the linear chromatic order.

    :param pitch_magnitudes: List of 12 pitch magnitudes in circle of fifths order
                             [C, G, D, A, E, B, F#, C#, G#, D#, A#, F]
    :return: List of 12 pitch magnitudes sorted back to chromatic order
             [C, C#, D, D#, E, F, F#, G, G#, A, A#, B]
    """

    # Circle of Fifths mapping indices
    circle_of_fifths_order = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]

    # Create an empty list to store the reordered values
    chromatic_magnitudes = [0] * 12

    # Rearrange pitch magnitudes back to chromatic order
    for chromatic_index, fifth_index in enumerate(circle_of_fifths_order):
        chromatic_magnitudes[fifth_index] = pitch_magnitudes[chromatic_index]

    return chromatic_magnitudes

def chroma_frequency_to_all_chords_similarity(frequencies, kernel=np.dot):
    """
    :param kernel: symmilarity measure between synthetic chord and non syntetic chord
    :param frequencies: array if size 12 with the chroma frequencies magnitudes
    :param k_chord: pitch of the major chord
    :return: a similarity measure between the chroma frequencies and the major chord
    """

    sorted_frequencies = sort_by_circle_of_fifths(frequencies)

    major_similarities = [chroma_frequency_to_chord_similarity(sorted_frequencies, k, "major", kernel) for k in range(12)]
    minor_similarities = [chroma_frequency_to_chord_similarity(sorted_frequencies, k, "minor", kernel) for k in range(12)]

    chord_templates = {
        # Major chords
        "C Major": major_similarities[0],  # C
        "C# Major": major_similarities[7],  # C#
        "D Major": major_similarities[2],  # D
        "D# Major": major_similarities[9],  # D#
        "E Major": major_similarities[4],  # E
        "F Major": major_similarities[11],  # F
        "F# Major": major_similarities[6],  # F#
        "G Major": major_similarities[1],  # G
        "G# Major": major_similarities[8],  # G#
        "A Major": major_similarities[3],  # A
        "A# Major": major_similarities[10],  # A#
        "B Major": major_similarities[5],  # B

        # Minor chords
        "C Minor": minor_similarities[0],  # Cm
        "C# Minor": minor_similarities[7],  # C#m
        "D Minor": minor_similarities[2],  # Dm
        "D# Minor": minor_similarities[9],  # D#m
        "E Minor": minor_similarities[4],  # Em
        "F Minor": minor_similarities[11],  # Fm
        "F# Minor": minor_similarities[6],  # F#m
        "G Minor": minor_similarities[1],  # Gm
        "G# Minor": minor_similarities[8],  # G#m
        "A Minor": minor_similarities[3],  # Am
        "A# Minor": minor_similarities[10],  # A#m
        "B Minor": minor_similarities[5],  # Bm
    }
    return chord_templates





