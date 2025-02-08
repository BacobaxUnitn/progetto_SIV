from utils.libs.constants import chord_templates
from utils.libs.chroma_creation import ChromaStudio
from utils.libs.constants import cosine_similarity

import pandas as pd


def similarity_scores(freq_array):
    scores = {}
    for chord_name, chord_template in chord_templates.items():
        scores[chord_name] = cosine_similarity(freq_array, chord_template)
    return scores


def chord_scores_list(chroma_studio: ChromaStudio, n=6):
    scores_list = []

    chroma = chroma_studio.get_median_smoothed_chroma()

    for i in range(chroma.shape[1]):
        scores = similarity_scores(chroma[:, i])
        scores["time"] = chroma_studio.idx_2_time(i)
        scores_list.append(scores)
    return scores_list


def create_chord_df(chroma_studio: ChromaStudio):

    scores_list = chord_scores_list(chroma_studio)

    return pd.DataFrame(scores_list)


def sample_df_per_beats(chroma_studio: ChromaStudio, df):
    period = chroma_studio.get_period()
    beat_duration = chroma_studio.get_beat_duration()

    how_many_samples_in_a_beat = int(beat_duration / period)
    return df.iloc[::how_many_samples_in_a_beat]


def filter_chord_df(df):
    # Identify the chord score columns (excluding the time column)
    chord_score_columns = df.columns[:-1]

    # Create a mask to identify the first occurrence of consecutive duplicate rows (excluding the time column)
    mask = df[chord_score_columns].ne(df[chord_score_columns].shift()).any(axis=1)

    # Apply the mask to filter the DataFrame
    return df[mask].reset_index(drop=True)


def clean_chord_score_from_chroma(chroma_studio: ChromaStudio):
    df = create_chord_df(chroma_studio)
    sampled_df = sample_df_per_beats(chroma_studio, df)
    return filter_chord_df(sampled_df)
