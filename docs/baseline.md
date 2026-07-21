# Baseline results (the bar every model has to beat)

Positive class = attack (1).

## DummyClassifier(strategy="most_frequent")

| split | accuracy | precision | recall | f1 | pr_auc |
|---|---|---|---|---|---|
| val | 0.5143 | 0.5143 | 1.0 | 0.6792 | 0.5143 |
| test | 0.5506 | 0.5506 | 1.0 | 0.7102 | 0.5506 |

## DummyClassifier(strategy="stratified")

| split | accuracy | precision | recall | f1 | pr_auc |
|---|---|---|---|---|---|
| val | 0.4968 | 0.5107 | 0.5127 | 0.5117 | 0.5125 |
| test | 0.5013 | 0.5505 | 0.5135 | 0.5313 | 0.5505 |

