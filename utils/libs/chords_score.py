import pandas as pd

from utils.libs.constants import cosine_similarity, similarity_scores
from utils.libs.plotting import plot_chord_heatmap
from utils.libs.spiral_model import chroma_frequency_to_all_chords_similarity


class ChordScores:
    """
    Class to compute and manage chord similarity scores based on chroma features.

    Attributes:
        chroma_matrix (np.ndarray): The chroma feature matrix (12 x num_time_frames).
        index_to_time (function): Function to convert index to time.
        period (float): Time period per frame.
        beat_duration (float): Duration of a beat.
        model (str): Chord similarity model to use ('base' or alternative model).
        kernel (function): Similarity function (default: cosine_similarity).
        scores_list (list): List containing chord similarity scores at each time step.
        df (pd.DataFrame): Processed DataFrame of chord similarity scores.
    """

    def __init__(self, chroma_matrix, index_to_time, period, beat_duration, model="base", kernel=cosine_similarity):
        """
        Initializes the ChordScores class and computes similarity scores.

        Parameters:
            chroma_matrix (np.ndarray): Chroma feature matrix.
            index_to_time (function): Function to convert index to time.
            period (float): Time period per frame.
            beat_duration (float): Duration of a beat.
            model (str, optional): Chord similarity model to use. Default is 'base'.
            kernel (function, optional): Similarity function. Default is cosine_similarity.
        """
        self.chroma_matrix = chroma_matrix
        self.kernel = kernel
        self.index_to_time = index_to_time
        self.period = period
        self.beat_duration = beat_duration
        self.model = model
        self.create_chord_scores_list(kernel=kernel)

        self.df = pd.DataFrame(self.scores_list)
        self.sample_df_per_beats()
        self.filter_chord_df()

    def create_chord_scores_list(self, kernel=None):
        """
        Computes similarity scores for each time frame in the chroma matrix.
        """
        self.scores_list = []

        for i in range(self.chroma_matrix.shape[1]):
            if self.model == "base":
                scores = similarity_scores(self.chroma_matrix[:, i], kernel=kernel)
            else:
                scores = chroma_frequency_to_all_chords_similarity(self.chroma_matrix[:, i], kernel=kernel)
            scores["time"] = self.index_to_time(i)
            self.scores_list.append(scores)

        return self

    def sample_df_per_beats(self):
        """
        Reduces the DataFrame sampling frequency to match beat duration.
        """
        how_many_samples_in_a_beat = int(self.beat_duration / self.period)
        self.df = self.df.iloc[::how_many_samples_in_a_beat]
        return self

    def filter_chord_df(self):
        """
        Removes consecutive duplicate chord scores to simplify the data.
        """
        chord_score_columns = self.df.columns[:-1]  # Exclude time column
        mask = self.df[chord_score_columns].ne(self.df[chord_score_columns].shift()).any(axis=1)
        self.df = self.df[mask].reset_index(drop=True)
        return self

    def plot_chord_heatmap(self, window: (int, int) = None):
        """
        Plots a heatmap of chord progression over time.

        Parameters:
            window (tuple, optional): Time range (start, end) for visualization.
        """
        plot_chord_heatmap(self.df, window)

    def get_df(self):
        """
        Returns the processed chord similarity DataFrame.

        Returns:
            pd.DataFrame: Chord similarity scores over time.
        """
        return self.df

    def get_scores_list(self):
        """
        Returns the list of raw chord similarity scores.

        Returns:
            list: Chord similarity scores at each time step.
        """
        return self.scores_list

    @staticmethod
    def extract_top_k(chord_timestamps, scores, k=2):
        """
        Extracts the top-k most similar chords from a given score dictionary.

        Parameters:
            chord_timestamps (list): List to store extracted chords.
            scores (dict): Dictionary containing similarity scores and time.
            k (int, optional): Number of top chords to extract. Default is 2.
        """
        scores_without_time = {key: val for key, val in scores.items() if key != "time"}
        if scores_without_time:
            sorted_chords = sorted(scores_without_time, key=scores_without_time.get, reverse=True)
            top_k_chords = sorted_chords[:k]
            chord_entry = {"time": scores["time"]}
            for i, chord in enumerate(top_k_chords):
                chord_entry[f"chord_{i + 1}"] = chord
            chord_timestamps.append(chord_entry)

    def get_top_chord_per_time(self, k=2):
        """
        Extracts the top-k chords at each time step from raw scores.

        Parameters:
            k (int, optional): Number of top chords to extract. Default is 2.

        Returns:
            list: List of dictionaries containing top chords at each time step.
        """
        chord_timestamps = []
        for scores in self.scores_list:
            self.extract_top_k(chord_timestamps, scores, k)
        return chord_timestamps

    def get_top_chord_per_time_from_df(self, k=2):
        """
        Extracts the top-k chords at each time step from the processed DataFrame.

        Parameters:
            k (int, optional): Number of top chords to extract. Default is 2.

        Returns:
            list: List of dictionaries containing top chords at each time step.
        """
        chord_timestamps = []
        for _, row in self.df.iterrows():
            self.extract_top_k(chord_timestamps, row, k)
        return chord_timestamps
