# Performance Metrics for Relevance Classifiers

| Classifier                                                                                   | Accuracy | F1   | Precision | Recall | Kappa |
|----------------------------------------------------------------------------------------------|----------|------|-----------|--------|-------|
| "mpox" or "monkeypox" in extracted keywords                                                  | 0.94     | 0.96 | 0.94      | 0.98   | 0.87  |
| "mpox", "monkeypox", "mpv", or "mpx" in extracted keywords                                   | 0.95     | 0.96 | 0.94      | 0.99   | 0.88  |
| 'monkeypox', 'transmission', or 'transmit' in extracted keywords                             | 0.94     | 0.96 | 0.94      | 0.98   | 0.86  |
| "briefing" not in title AND 'monkeypox', 'transmission', or 'transmit' in extracted keywords | 0.94     | 0.96 | 0.94      | 0.98   | 0.86  |
| Naive Bayes on extracted keywords                                                            | 0.87     | 0.91 | 0.84      | 0.99   | 0.67  |
| Naive Bayes on text                                                                          | 0.93     | 0.95 | 0.91      | 0.99   | 0.82  |
| Naive Bayes on title                                                                         | 0.8      | 0.87 | 0.78      | 0.99   | 0.44  |
| Naive Bayes on text AND Naive Bayes on title agree                                           | 0.93     | 0.95 | 0.93      | 0.98   | 0.84  |
| BERT on title + text; 60/20/20 train/val/test split                                          | 0.96     | 0.97 | 0.96      | 0.99   | 0.91  |
| DeBERTa on title + text; 60/20/20 train/val/test split                                       | 0.98     | 0.99 | 0.99      | 0.99   | 0.95  |
