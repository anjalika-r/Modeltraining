"""Reproducible recording-level splits within selected participants."""
import pandas as pd
from sklearn.model_selection import train_test_split


def split_selected_participants(metadata, participants, validation_fraction, seed, previous_manifest=None):
    if not 0 < validation_fraction < 1:
        raise ValueError('Validation fraction must be between zero and one')
    unknown = set(participants) - set(metadata['participant'])
    if unknown:
        raise ValueError(f'Unknown participants: {sorted(unknown)}')
    selected = metadata.loc[metadata['participant'].isin(participants)].copy()
    if selected['source_file'].isna().any():
        raise ValueError('Missing source_file: cannot keep recordings together')
    if (selected.groupby(['participant', 'source_file'])['class_id'].nunique() != 1).any():
        raise ValueError('A source recording has conflicting labels')
    if previous_manifest is not None:
        keys = ['participant', 'source_file']
        previous = previous_manifest.loc[previous_manifest['split'].isin(['train', 'validation'])]
        if previous.duplicated(keys).any():
            raise ValueError('Previous manifest has duplicate recording keys')
        lookup = previous.set_index(keys)['split']
        assignments = pd.MultiIndex.from_frame(metadata[keys]).map(lookup)
        known = pd.Series(assignments, index=metadata.index)
        if not pd.MultiIndex.from_frame(previous[keys]).isin(pd.MultiIndex.from_frame(selected[keys])).all():
            raise ValueError('Previous train/validation recordings are missing from selected data')
        joined = selected.merge(previous[keys + ['class_label']], on=keys, suffixes=('', '_previous'))
        if not (joined['class_label'] == joined['class_label_previous']).all():
            raise ValueError('Previous recording labels changed')
        new = selected.loc[known.loc[selected.index].isna()]
        train = known.eq('train')
        val = known.eq('validation')
        if len(new):
            new_train, new_val = split_selected_participants(new, new.participant.unique(), validation_fraction, seed)
            train.loc[new.index] = new_train
            val.loc[new.index] = new_val
        return train, val
    # source_file identifies the original recording in this dataset. Repeated
    # rows from the same source stay together; clipNumber alone is not unique.
    keys = ['participant', 'source_file']
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
