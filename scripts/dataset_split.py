"""Reproducible recording-level splits within selected participants."""
import pandas as pd
from sklearn.model_selection import train_test_split


def split_selected_participants(metadata, participants, validation_fraction, seed):
    if not 0 < validation_fraction < 1:
        raise ValueError('Validation fraction must be between zero and one')
    unknown = set(participants) - set(metadata['participant'])
    if unknown:
        raise ValueError(f'Unknown participants: {sorted(unknown)}')
    selected = metadata.loc[metadata['participant'].isin(participants)].copy()
    # source_file identifies the original recording in this dataset. Repeated
    # rows from the same source stay together; clipNumber alone is not unique.
    if selected['source_file'].isna().any():
        raise ValueError('Missing source_file: cannot keep recordings together')
    keys = ['participant', 'source_file']
    if (selected.groupby(keys)['class_id'].nunique() != 1).any():
        raise ValueError('A source recording has conflicting labels')
    recordings = selected.drop_duplicates(keys)
    strata = recordings['participant'].astype(str) + ':' + recordings['class_id'].astype(str)
    if strata.value_counts().min() < 2:
        raise ValueError('Each participant/letter needs at least two independent recordings')
    _, validation = train_test_split(recordings, test_size=validation_fraction,
                                    random_state=seed, stratify=strata)
    validation_keys = pd.MultiIndex.from_frame(validation[keys])
    selected_mask = metadata['participant'].isin(participants)
    val_mask = pd.Series(pd.MultiIndex.from_frame(metadata[keys]).isin(validation_keys), index=metadata.index)
    return selected_mask & ~val_mask, val_mask
