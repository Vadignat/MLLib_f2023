import numpy as np
from datasets.sportsmans_height import Sportsmanheight
from models.simple_classifier import Classifier

from utils.metrics import accuracy, precision, recall, f1_score, TruePositives, FalsePositives, TrueNegatives, \
    FalseNegatives
from utils.visualisation import Visualisation

dataset = Sportsmanheight()()
predictions = Classifier()(dataset['height'])
gt = dataset['class']

accuracy_values = []
precision_values = []
recall_values = []
f1_score_values = []
thresholds = [.6, .7, .73, .75, .76, .77, .78, .79, .8, .81, .82, .824]
true_positives = []
false_positives = []
true_negatives = []
false_negatives = []

for i in range(len(thresholds)):
    predictions_i = predictions >= thresholds[i]
    tp = TruePositives(predictions_i, gt)
    fp = FalsePositives(predictions_i, gt)
    tn = TrueNegatives(predictions_i, gt)
    fn = FalseNegatives(predictions_i, gt)

    true_positives.append(tp)
    false_positives.append(fp)
    true_negatives.append(tn)
    false_negatives.append(fn)
    accuracy_values.append(accuracy(predictions_i, gt))
    precision_values.append(precision(predictions_i, gt))
    recall_values.append(recall(predictions_i, gt))
    f1_score_values.append(f1_score(predictions_i, gt))

print(true_positives)
print(true_negatives)
print(false_positives)
print(false_negatives)
print(recall_values)
print(precision_values)
vis = Visualisation()
fig = vis.plot_precision_recall_curve(thresholds, precision_values, recall_values, accuracy_values, f1_score_values)
fig.show()

