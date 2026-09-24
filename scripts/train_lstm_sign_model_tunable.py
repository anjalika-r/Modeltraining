#!/usr/bin/env python3
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sign_features import FEATURE_DIM, SCHEMA, MODEL_SCHEMAS, model_features, load_class_names
from dataset_split import split_selected_participants


def parse_args():
    p = argparse.ArgumentParser(description='Train a tunable LSTM sign classifier.')
    p.add_argument('--data-dir', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--input-file', default='sequences_body_hands.npy')
    p.add_argument('--feature-set', choices=MODEL_SCHEMAS, default='shape-contact',
                   help='Derived hand geometry and cross-hand XY features, or original baseline')
    p.add_argument('--val-participant', default='rithika')
    p.add_argument('--test-participant', help='Optional untouched participant for final evaluation')
    p.add_argument('--participants', nargs='+', help='Use only these participants, splitting their recordings into train/validation')
    p.add_argument('--validation-fraction', type=float, default=0.15)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--learning-rate', type=float, default=5e-4)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--lr-patience', type=int, default=6)
    p.add_argument('--lr-factor', type=float, default=0.5)
    p.add_argument('--min-lr', type=float, default=1e-6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--frame-dense-units', type=int, default=128)
    p.add_argument('--lstm-units', type=int, default=64)
    p.add_argument('--dense-units', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.40)
    p.add_argument('--recurrent-dropout', type=float, default=0.0)
    p.add_argument('--bidirectional', action='store_true')
    p.add_argument('--l2', type=float, default=0.0)
    args = p.parse_args()
    if args.participants and args.test_participant:
        p.error('--participants excludes everyone else; do not combine with --test-participant')
    if not 0 < args.validation_fraction < 1:
        p.error('--validation-fraction must be between zero and one')
    if min(args.epochs, args.batch_size, args.frame_dense_units, args.lstm_units, args.dense_units) < 1:
        p.error('Epochs, batch size and layer sizes must be positive')
    if args.learning_rate <= 0 or args.min_lr <= 0 or not 0 < args.lr_factor < 1:
        p.error('Learning rates must be positive; lr-factor must lie between 0 and 1')
    if args.patience < 0 or args.lr_patience < 0 or args.l2 < 0 or not 0 <= args.dropout < 1 or not 0 <= args.recurrent_dropout < 1:
        p.error('Invalid patience, regularization, or dropout')
    return args


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


def build_model(t, f, n_classes, args):
    reg = tf.keras.regularizers.l2(args.l2) if args.l2 > 0 else None
    inp = tf.keras.Input(shape=(t, f), name='sequence')
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(args.frame_dense_units, activation='relu', kernel_regularizer=reg),
        name='frame_projection')(inp)
    x = tf.keras.layers.Dropout(args.dropout)(x)
    lstm = tf.keras.layers.LSTM(
        args.lstm_units,
        recurrent_dropout=args.recurrent_dropout,
        kernel_regularizer=reg,
        recurrent_regularizer=reg)
    x = tf.keras.layers.Bidirectional(lstm)(x) if args.bidirectional else lstm(x)
    x = tf.keras.layers.Dropout(args.dropout)(x)
    x = tf.keras.layers.Dense(args.dense_units, activation='relu', kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(args.dropout)(x)
    out = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


def save_curves(hist, out):
    plt.figure(figsize=(8,5)); plt.plot(hist['accuracy'], label='Train'); plt.plot(hist['val_accuracy'], label='Validation')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(out/'accuracy_curve.png', dpi=160); plt.close()
    plt.figure(figsize=(8,5)); plt.plot(hist['loss'], label='Train'); plt.plot(hist['val_loss'], label='Validation')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(out/'loss_curve.png', dpi=160); plt.close()


def main():
    args = parse_args(); set_seed(args.seed)
    data_dir = args.data_dir.expanduser().resolve(); out = args.output_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out / 'run_config.json').exists():
        raise SystemExit('Output already contains a run. Choose a new --output-dir.')
    X = np.load(data_dir / args.input_file)
    y = np.load(data_dir / 'labels.npy')
    meta = pd.read_csv(data_dir / 'metadata.csv')
    if len(X) != len(y) or len(X) != len(meta): raise SystemExit('X/y/metadata row count mismatch')
    if not np.isfinite(X).all(): raise SystemExit('X contains NaN or inf')
    if X.ndim != 3 or X.shape[1] < 2 or X.shape[2] != FEATURE_DIM:
        raise SystemExit('Expected normalized body/hand tensor (samples, frames, 227)')
    X = X.astype(np.float32)
    if meta['participant'].isna().any(): raise SystemExit('Missing participant IDs')
    class_names = load_class_names(data_dir, y); n_classes = len(class_names)
    if 'class_id' not in meta or not np.array_equal(meta['class_id'].to_numpy(), y):
        raise SystemExit('Metadata class IDs disagree with labels.npy')
    if 'class_label' not in meta or not np.array_equal(meta['class_label'].to_numpy(), np.asarray(class_names)[y]):
        raise SystemExit('Metadata class labels disagree with mapping')
    pose = X[..., :99].reshape(*X.shape[:2], 33, 3)
    if not np.allclose((pose[..., 11, :] + pose[..., 12, :]) / 2, 0, atol=1e-4) or not np.allclose(np.linalg.norm(pose[..., 11, :2] - pose[..., 12, :2], axis=-1), 1, atol=1e-4):
        raise SystemExit('Input does not match shoulder-normalized feature contract')
    schema = MODEL_SCHEMAS[args.feature_set]
    X = model_features(X, schema)
    part = meta['participant'].astype(str)
    if args.participants:
        train_mask, val_mask = split_selected_participants(meta, args.participants, args.validation_fraction, args.seed)
        test_mask = pd.Series(False, index=meta.index)
        args.val_participant = None
    else:
        if args.val_participant not in set(part): raise SystemExit(f'Unknown participant: {args.val_participant}')
        if args.test_participant and (args.test_participant not in set(part) or args.test_participant == args.val_participant):
            raise SystemExit('Test participant must exist and differ from validation participant')
        val_mask = part == args.val_participant
        test_mask = part == args.test_participant
        train_mask = ~(val_mask | test_mask)
    X_train, y_train = X[train_mask.to_numpy()], y[train_mask.to_numpy()]
    X_val, y_val = X[val_mask.to_numpy()], y[val_mask.to_numpy()]
    missing = sorted(set(range(n_classes)) - set(np.unique(y_train)))
    if missing: raise SystemExit('Validation classes missing from training: ' + ', '.join(class_names[i] for i in missing))
    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, cw)}

    config = vars(args).copy(); config['data_dir'] = str(data_dir); config['output_dir'] = str(out); config['input_shape'] = list(X.shape)
    with (out/'run_config.json').open('w') as f: json.dump(config, f, indent=2)
    durations = []
    for value in meta.loc[train_mask, 'selected_timestamps']:
        timestamps = json.loads(value)
        if len(timestamps) != X.shape[1] or np.any(np.diff(timestamps) <= 0):
            raise SystemExit('Invalid selected_timestamps in training metadata')
        durations.append((timestamps[-1] - timestamps[0]) / 1000)
    contract = {
        'schema': schema, 'capture_schema': SCHEMA,
        'feature_set': args.feature_set,
        'timesteps': int(X.shape[1]), 'feature_dim': int(X.shape[2]),
        'window_seconds': float(np.median(durations)),
        'sampling': 'uniform_frame_indices_floor',
        'landmarks': 'image_normalized_xyz; pose33,left_hand21,right_hand21,left_present,right_present',
        'normalization': 'subtract shoulder XYZ midpoint; divide XYZ by shoulder XY distance; absent hands zero',
        'mirror_input': False,
        'mirror_note': 'Original capture mirroring is undocumented; compare with capture application.',
        'class_names': class_names,
    }
    (out/'inference_config.json').write_text(json.dumps(contract, indent=2))
    (out/'class_mapping.json').write_text((data_dir/'class_mapping.json').read_text())
    split_meta = meta.copy()
    split_meta['split'] = np.select([train_mask, val_mask, test_mask], ['train', 'validation', 'test'], default='excluded')
    split_meta.to_csv(out/'split_manifest.csv', index=False)

    print('X shape:', X.shape)
    print('Training:', X_train.shape)
    print('Validation:', X_val.shape)
    print('\nTraining participants:')
    print(meta.loc[train_mask, 'participant'].value_counts())
    print('\nValidation participants:')
    print(meta.loc[val_mask, 'participant'].value_counts())
    print('\nHyperparameters:')
    print(json.dumps({k: config[k] for k in ['frame_dense_units','lstm_units','dense_units','dropout','recurrent_dropout','bidirectional','l2','learning_rate','batch_size','epochs','patience']}, indent=2))

    model = build_model(X.shape[1], X.shape[2], n_classes, args)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(out/'best_lstm_model.keras', monitor='val_loss', save_best_only=True, verbose=1),
    ]

    history_obj = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=True,
        verbose=1)

    # Explicitly reload the selected checkpoint even when training reaches its epoch limit.
    model = tf.keras.models.load_model(out/'best_lstm_model.keras')
    model.save(out/'final_lstm_model.keras')
    hist = pd.DataFrame(history_obj.history)
    hist.index = np.arange(1, len(hist)+1); hist.index.name='epoch'; hist.to_csv(out/'training_history.csv')
    save_curves(hist, out)

    val_loss, val_acc = model.evaluate(X_val, y_val, batch_size=args.batch_size, verbose=0)
    probs = model.predict(X_val, batch_size=args.batch_size, verbose=0)
    pred = np.argmax(probs, axis=1)
    report = classification_report(y_val, pred, labels=np.arange(n_classes), target_names=class_names, zero_division=0, digits=4)
    (out/'classification_report.txt').write_text(report)

    cm = confusion_matrix(y_val, pred, labels=np.arange(n_classes))
    plt.figure(figsize=(18,16)); plt.imshow(cm, aspect='auto'); plt.title('Validation confusion matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
    ticks=np.arange(n_classes); plt.xticks(ticks,class_names,rotation=90,fontsize=7); plt.yticks(ticks,class_names,fontsize=7); plt.colorbar(); plt.tight_layout(); plt.savefig(out/'confusion_matrix.png', dpi=180); plt.close()

    pred_df = meta.loc[val_mask].copy().reset_index(drop=True)
    pred_df['true_class_id']=y_val; pred_df['predicted_class_id']=pred
    pred_df['true_class']=[class_names[i] for i in y_val]; pred_df['predicted_class']=[class_names[i] for i in pred]
    pred_df['confidence']=probs.max(axis=1); pred_df['correct']=pred==y_val
    pred_df.to_csv(out/'validation_predictions.csv', index=False)

    best_epoch = int(hist['val_loss'].idxmin())
    summary = {
        'input_shape': list(X.shape), 'num_classes': n_classes,
        'training_sequences': int(len(X_train)), 'validation_sequences': int(len(X_val)),
        'validation_participant': args.val_participant,
        'split_strategy': 'within_participant_recordings' if args.participants else 'held_out_participants',
        'validation_participants': sorted(meta.loc[val_mask, 'participant'].unique().tolist()),
        'excluded_sequences': int((~(train_mask | val_mask | test_mask)).sum()),
        'best_epoch': best_epoch,
        'best_epoch_val_loss': float(hist.loc[best_epoch,'val_loss']),
        'best_epoch_val_accuracy': float(hist.loc[best_epoch,'val_accuracy']),
        'restored_model_validation_loss': float(val_loss),
        'restored_model_validation_accuracy': float(val_acc),
        'epochs_completed': int(len(hist)),
        'hyperparameters': config,
    }
    with (out/'evaluation_summary.json').open('w') as f: json.dump(summary, f, indent=2)
    if test_mask.any():
        X_test, y_test = X[test_mask.to_numpy()], y[test_mask.to_numpy()]
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        test_pred = model.predict(X_test, verbose=0).argmax(axis=1)
        (out/'test_classification_report.txt').write_text(classification_report(
            y_test, test_pred, labels=np.arange(n_classes), target_names=class_names, zero_division=0, digits=4))
        (out/'test_summary.json').write_text(json.dumps({
            'participant': args.test_participant, 'sequences': len(y_test),
            'loss': float(test_loss), 'accuracy': float(test_accuracy),
        }, indent=2))

    print('\nDONE')
    print('Best epoch:', best_epoch)
    print('Best epoch val accuracy:', round(summary['best_epoch_val_accuracy'],4))
    print('Restored model val accuracy:', round(val_acc,4))
    print('Outputs saved to:', out)


if __name__ == '__main__':
    main()
