from baseline import Baseline
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

baseline = Baseline(None)
X = []
y = []
for path in Path(r"data\annotation_edited").rglob('*.json'):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        X.append(data['emoted_caption'])
        y.append(data['emotion'])


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 42, shuffle = True, stratify = y)
fitted_model = baseline.fit(X_train, y_train)

scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
scores = cross_validate(fitted_model.svc, fitted_model._Baseline__vectors_merge(fitted_model._Baseline__sentence_preprocessing(X)), y, cv = 5, scoring=scoring)
print(f"Accuracy:  {scores['test_accuracy'].mean():.3f}")
print(f"Precision: {scores['test_precision_macro'].mean():.3f}")
print(f"Recall:    {scores['test_recall_macro'].mean():.3f}")
print(f"F1-score:  {scores['test_f1_macro'].mean():.3f}")