"""Training-only weights that give each signer and each of their letters equal mass."""
import numpy as np


def signer_letter_weights(metadata):
    if len(metadata) == 0 or metadata[['participant', 'class_id']].isna().any().any():
        raise ValueError('Nonempty training metadata with signer and class IDs is required')
    counts = metadata.groupby(['participant', 'class_id'])['class_id'].transform('size')
    letters = metadata.groupby('participant')['class_id'].transform('nunique')
    weights = len(metadata) / (metadata.participant.nunique() * letters * counts)
    return weights.to_numpy(dtype=np.float32)
