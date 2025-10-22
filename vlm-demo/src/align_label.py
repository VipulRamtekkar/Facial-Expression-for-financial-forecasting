import pandas as pd


def align_frames_with_labels(scored_rows, labels):
    df = pd.DataFrame(scored_rows)
    for key, value in labels.items():
        df[key] = value
    return df
