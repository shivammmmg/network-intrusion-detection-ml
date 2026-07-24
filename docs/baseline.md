# Baseline results (the bar every model has to beat)

Positive class = attack (1).

## DummyClassifier(strategy="most_frequent")

| split | accuracy | precision | recall | f1 | pr_auc |
|---|---|---|---|---|---|
| val | 0.5157 | 0.0 | 0.0 | 0.0 | 0.4843 |
| test | 0.4494 | 0.0 | 0.0 | 0.0 | 0.5506 |

## DummyClassifier(strategy="stratified")

| split | accuracy | precision | recall | f1 | pr_auc |
|---|---|---|---|---|---|
| val | 0.4945 | 0.478 | 0.4752 | 0.4766 | 0.4813 |
| test | 0.4985 | 0.5506 | 0.4849 | 0.5157 | 0.5506 |

