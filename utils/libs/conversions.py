

def get_index_time_mapping(hop_length, sr):
    def time_to_index(time):
        """
        Converts time to index.

        Parameters:
        - time (float): Time in seconds.
        - sr (int): Sampling rate of the audio.
        - hop_length (int): Hop size used in STFT.

        Returns:
        - int: Index corresponding to the time.
        """
        time_per_column = hop_length / sr
        return int(time / time_per_column)

    def index_to_time(index):
        """
        Converts index to time.

        Parameters:
        - index (int): Index.
        - sr (int): Sampling rate of the audio.
        - hop_length (int): Hop size used in STFT.

        Returns:
        - float: Time in seconds corresponding to the index.
        """
        time_per_column = hop_length / sr
        return index * time_per_column

    def cut_time(chroma_matrix, start_time, end_time):
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
        start_index = time_to_index(start_time)
        end_index = time_to_index(end_time)

        # Ensure indices are within bounds
        start_index = max(0, start_index)
        end_index = min(chroma_matrix.shape[1], end_index)

        # Slice the chroma matrix
        return chroma_matrix[:, start_index:end_index]

    return time_to_index, index_to_time, cut_time
