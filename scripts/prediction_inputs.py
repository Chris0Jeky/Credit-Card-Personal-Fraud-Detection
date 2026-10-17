"""Validate complete prediction coverage and bind CSV outputs to their inputs."""

import hashlib
import json

import numpy as np
import pandas as pd

import config


def input_digest(paths):
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    return hashlib.sha256(json.dumps(hashes).encode()).hexdigest()


def simulation_digest(transactions_file=None):
    return input_digest([transactions_file or config.TRANSACTIONS_FILE,
                         config.ACCOUNT_DETAILS_FILE])


def write_provenance(output_file, source_digest):
    """Record the input snapshot captured before processing, plus the saved CSV digest."""
    provenance = {
        'input_sha256': source_digest,
        'output_sha256': hashlib.sha256(output_file.read_bytes()).hexdigest(),
    }
    output_file.with_suffix(output_file.suffix + '.provenance.json').write_text(
        json.dumps(provenance, indent=4), encoding='utf-8')


def verify_provenance(output_file, source_digest):
    provenance_file = output_file.with_suffix(output_file.suffix + '.provenance.json')
    if not provenance_file.exists():
        raise ValueError(f"{output_file.name}: missing provenance; regenerate this output from current inputs.")
    provenance = json.loads(provenance_file.read_text(encoding='utf-8'))
    if provenance.get('input_sha256') != source_digest:
        raise ValueError(f"{output_file.name}: stale input digest; regenerate this output from current inputs.")
    if provenance.get('output_sha256') != hashlib.sha256(output_file.read_bytes()).hexdigest():
        raise ValueError(f"{output_file.name}: output digest mismatch; regenerate this output.")


def validate_rows(frame, name, expected_ids=None, numeric_columns=()):
    required = ['trans_num', *numeric_columns]
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{name}: missing required columns {missing_columns}.")
    if frame['trans_num'].isna().any():
        raise ValueError(f"{name}: null trans_num.")
    if frame['trans_num'].duplicated().any():
        raise ValueError(f"{name}: duplicate trans_num.")
    if expected_ids is not None:
        actual_ids = set(frame['trans_num'])
        coverage = f"{name} coverage: {len(expected_ids & actual_ids)}/{len(expected_ids)}"
        print(f"   {coverage}")
        if actual_ids != expected_ids:
            raise ValueError(f"{coverage}; missing_prediction: {sorted(expected_ids - actual_ids)}; "
                             f"unexpected trans_num: {sorted(actual_ids - expected_ids)}.")
    if not np.isfinite(frame[list(numeric_columns)].apply(pd.to_numeric, errors='coerce')).all().all():
        raise ValueError(f"{name}: missing or invalid prediction/label values.")
    for column in ('is_fraud', 'rule_triggered', 'anomaly', 'predicted_fraud'):
        if column in numeric_columns and not frame[column].isin([0, 1]).all():
            raise ValueError(f"{name}: missing or invalid binary values in {column}.")


def load_prediction_inputs():
    source_digest = simulation_digest()
    paths = [config.FLAGGED_TRANSACTIONS_FILE, config.PCA_PREDICTIONS_FILE,
             config.RF_PREDICTIONS_FILE]
    assessment_digest = input_digest([config.TRANSACTIONS_FILE, config.ACCOUNT_DETAILS_FILE, *paths])
    transactions = pd.read_csv(config.TRANSACTIONS_FILE, dtype={'trans_num': str})
    validate_rows(transactions, 'Transactions')
    if transactions.empty:
        raise ValueError('No transactions to assess.')
    expected_ids = set(transactions['trans_num'])
    frames = [pd.read_csv(path, dtype={'trans_num': str}) for path in paths]
    errors = []
    for frame, name, columns in zip(frames, ('Rules', 'PCA', 'RF'),
                                    (('is_fraud', 'rule_triggered'), ('anomaly', 'anomaly_score'),
                                     ('predicted_fraud', 'fraud_probability'))):
        try:
            validate_rows(frame, name, expected_ids, columns)
        except ValueError as error:
            errors.append(str(error))
    if errors:
        raise ValueError('; '.join(errors))
    for path in paths:
        verify_provenance(path, source_digest)
    return *frames, assessment_digest
