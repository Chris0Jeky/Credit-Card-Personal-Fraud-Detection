import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import config
from scripts import evaluate_performance, integrate_results


class PredictionCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        names = {
            'TRANSACTIONS_FILE': 'transactions.csv',
            'ACCOUNT_DETAILS_FILE': 'account.json',
            'FLAGGED_TRANSACTIONS_FILE': 'flags.csv',
            'PCA_PREDICTIONS_FILE': 'pca.csv',
            'RF_PREDICTIONS_FILE': 'rf.csv',
            'INTEGRATED_ASSESSMENT_FILE': 'integrated.csv',
            'EVALUATION_REPORT_FILE': 'evaluation.json',
        }
        self.addCleanup(patch.stopall)
        for name, filename in names.items():
            patch.object(config, name, root / filename).start()
        self.transactions = pd.DataFrame({
            'trans_num': ['BENIGN_1', 'FRAUD_1'],
            'trans_date_trans_time': ['2026-01-01 10:00:00', '2026-01-01 11:00:00'],
            'amt': [10.0, 20.0], 'merchant': ['shop', 'shop'],
            'category': ['shopping', 'shopping'], 'is_fraud': [0, 1],
        })
        self.transactions.to_csv(config.TRANSACTIONS_FILE, index=False)
        config.ACCOUNT_DETAILS_FILE.write_text('{}', encoding='utf-8')
        self.flags = self.transactions.assign(rule_flags='', rule_triggered=0)
        self.pca = self.transactions.assign(anomaly=[0, 1], anomaly_score=[0.01, -0.1])
        self.rf = self.transactions.assign(predicted_fraud=[0, 1], fraud_probability=[0.0, 1.0])
        self.write_inputs()

    def source_paths(self):
        return [config.TRANSACTIONS_FILE, config.ACCOUNT_DETAILS_FILE]

    def assessment_paths(self):
        return self.source_paths() + [config.FLAGGED_TRANSACTIONS_FILE,
                                      config.PCA_PREDICTIONS_FILE, config.RF_PREDICTIONS_FILE]

    def write_output(self, path, frame, inputs=None):
        frame.to_csv(path, index=False)
        hashes = [hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (inputs or self.source_paths())]
        provenance = {
            'input_sha256': hashlib.sha256(json.dumps(hashes).encode()).hexdigest(),
            'output_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        path.with_suffix(path.suffix + '.provenance.json').write_text(
            json.dumps(provenance), encoding='utf-8')

    def write_inputs(self):
        for path, frame in [(config.FLAGGED_TRANSACTIONS_FILE, self.flags),
                            (config.PCA_PREDICTIONS_FILE, self.pca),
                            (config.RF_PREDICTIONS_FILE, self.rf)]:
            self.write_output(path, frame)

    def run_main(self, module):
        output = io.StringIO()
        with patch('sys.argv', ['test']), contextlib.redirect_stdout(output), \
                contextlib.redirect_stderr(output):
            try:
                result = module.main()
            except SystemExit as error:
                result = error.code
        return result, output.getvalue()

    def assert_refused(self, message):
        for module in (integrate_results, evaluate_performance):
            with self.subTest(module=module.__name__):
                result, output = self.run_main(module)
                self.assertEqual(result, 1, output)
                self.assertIn(message, output)
        self.assertFalse(config.INTEGRATED_ASSESSMENT_FILE.exists())
        self.assertFalse(config.EVALUATION_REPORT_FILE.exists())

    def test_missing_predictions_refuse_scoring_and_evaluation(self):
        for missing in ('pca', 'rf', 'both'):
            with self.subTest(missing=missing):
                self.write_output(config.PCA_PREDICTIONS_FILE,
                                  self.pca.iloc[:1] if missing in ('pca', 'both') else self.pca)
                self.write_output(config.RF_PREDICTIONS_FILE,
                                  self.rf.iloc[:1] if missing in ('rf', 'both') else self.rf)
                for module in (integrate_results, evaluate_performance):
                    result, output = self.run_main(module)
                    self.assertEqual(result, 1, output)
                    self.assertIn('coverage: 1/2', output)
                    self.assertIn('missing_prediction', output)
                    self.assertIn('FRAUD_1', output)
                    self.assertFalse(config.INTEGRATED_ASSESSMENT_FILE.exists())
                    self.assertFalse(config.EVALUATION_REPORT_FILE.exists())

    def test_duplicate_transaction_keys_refused(self):
        for name in ('FLAGGED_TRANSACTIONS_FILE', 'PCA_PREDICTIONS_FILE', 'RF_PREDICTIONS_FILE'):
            with self.subTest(input=name):
                self.write_inputs()
                path = getattr(config, name)
                frame = pd.read_csv(path)
                self.write_output(path, pd.concat([frame, frame.iloc[:1]]))
                self.assert_refused('duplicate trans_num')

    def test_null_transaction_keys_refused(self):
        self.pca.loc[0, 'trans_num'] = None
        self.write_inputs()
        self.assert_refused('null trans_num')

    def test_null_model_values_refused(self):
        for name, column in (('PCA_PREDICTIONS_FILE', 'anomaly_score'),
                             ('RF_PREDICTIONS_FILE', 'fraud_probability')):
            with self.subTest(column=column):
                self.write_inputs()
                path = getattr(config, name)
                frame = pd.read_csv(path)
                frame.loc[1, column] = float('nan')
                self.write_output(path, frame)
                self.assert_refused('missing or invalid')

    def test_stale_predictions_with_same_ids_refused(self):
        self.transactions.loc[1, 'amt'] = 200.0
        self.transactions.to_csv(config.TRANSACTIONS_FILE, index=False)
        self.assert_refused('stale')

    def test_changed_prediction_file_refused(self):
        self.pca.loc[1, 'anomaly_score'] = 0.01
        self.pca.to_csv(config.PCA_PREDICTIONS_FILE, index=False)
        self.assert_refused('digest')

    def test_untracked_legacy_predictions_refused(self):
        config.PCA_PREDICTIONS_FILE.with_suffix('.csv.provenance.json').unlink()
        self.assert_refused('provenance')

    def test_complete_matching_inputs_preserve_scores_and_metrics(self):
        result, output = self.run_main(integrate_results)
        self.assertEqual(result, 0, output)
        assessment = pd.read_csv(config.INTEGRATED_ASSESSMENT_FILE)
        self.assertEqual(assessment['risk_score'].tolist(), [0.0, 0.7])
        self.assertEqual(assessment['risk_level'].tolist(), ['Low', 'High'])
        result, output = self.run_main(evaluate_performance)
        self.assertEqual(result, 0, output)
        report = json.loads(config.EVALUATION_REPORT_FILE.read_text())
        self.assertEqual(report['rules']['confusion_matrix'], {'tn': 1, 'fp': 0, 'fn': 1, 'tp': 0})
        for name in ('pca_isolation_forest', 'random_forest', 'integrated_score'):
            self.assertEqual(report[name]['confusion_matrix'], {'tn': 1, 'fp': 0, 'fn': 0, 'tp': 1})
            for metric in ('accuracy', 'precision', 'recall', 'f1_score', 'auprc'):
                self.assertEqual(report[name][metric], 1.0)

    def test_absent_optional_integrated_results_not_evaluated(self):
        result, output = self.run_main(evaluate_performance)
        self.assertEqual(result, 0, output)
        report = json.loads(config.EVALUATION_REPORT_FILE.read_text())
        self.assertNotIn('integrated_score', report)

    def test_partial_integrated_results_refused(self):
        integrated = self.flags.iloc[:1].assign(risk_score=0.0, risk_level='Low')
        self.write_output(config.INTEGRATED_ASSESSMENT_FILE, integrated, self.assessment_paths())
        result, output = self.run_main(evaluate_performance)
        self.assertEqual(result, 1, output)
        self.assertIn('coverage: 1/2', output)
        self.assertFalse(config.EVALUATION_REPORT_FILE.exists())

    def test_integrated_results_bound_to_prediction_files(self):
        result, output = self.run_main(integrate_results)
        self.assertEqual(result, 0, output)
        self.rf.loc[1, 'fraud_probability'] = 0.9
        self.write_output(config.RF_PREDICTIONS_FILE, self.rf)
        result, output = self.run_main(evaluate_performance)
        self.assertEqual(result, 1, output)
        self.assertIn('stale', output)
        self.assertFalse(config.EVALUATION_REPORT_FILE.exists())


if __name__ == '__main__':
    unittest.main()
