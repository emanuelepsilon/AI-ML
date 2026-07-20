# Evaluation Report

The full pipeline was run on a deterministic synthetic dataset.

| Measure | Result |
| --- | ---: |
| Invoice extraction field accuracy | 1.000 |
| Invoice category accuracy | 1.000 |
| Reconciliation precision | 1.000 |
| Reconciliation recall | 1.000 |
| Anomaly precision | 0.727 |
| Anomaly recall | 1.000 |
| Assistant citation validity | 1.000 |
| Assistant source use | 1.000 |
| Unsupported question refusal | 1.000 |

Dataset: 108 invoices and 110 transactions. The labels contain 84 payment matches and 8 anomalies.

The figures measure this fixed demonstration dataset. They are not estimates of performance on real financial records.
