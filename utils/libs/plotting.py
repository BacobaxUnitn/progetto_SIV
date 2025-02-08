import matplotlib.pyplot as plt
import librosa
import seaborn as sns


def display_chromagram(chroma):
    plt.figure(figsize=(20, 4))
    librosa.display.specshow(chroma, x_axis="time", y_axis="chroma", cmap="coolwarm")
    plt.colorbar(label="Magnitude")
    plt.title("Chromagram (Using Custom chroma_stft Function)")
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch Class")
    plt.show()


def plot_chord_heatmap(df, window: (int, int) = None):
    time_df = df.set_index("time", inplace=False)

    if window is not None:
        start, end = window
        time_df = time_df.loc[start:end]

    plt.figure(figsize=(20, 10))  # Set figure size
    sns.heatmap(time_df.T, cmap="coolwarm", annot=False, linewidths=1)

    # Add labels and title
    plt.xlabel("Time")
    plt.ylabel("Chords")
    plt.title("Chord Progression Heatmap")

    plt.show()
