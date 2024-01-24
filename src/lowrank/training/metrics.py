import torch
from datetime import datetime
import time

# def precision_recall_f1(y_pred, y_true, num_classes=10):
#         with torch.no_grad():
#             _, predictions = torch.max(y_pred, 1)
#             precision_sum, recall_sum, f1_sum = 0, 0, 0

#             for i in range(num_classes):
#                 true_positive = ((predictions == i) and (y_true == i)).sum().item()
#                 false_positive = ((predictions == i) and (y_pred != i)).sum().item()
#                 false_negative = ((predictions != i) and (y_pred == i)).sum().item()

#                 precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
#                 recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
#                 f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

#                 precision_sum += precision
#                 recall_sum += recall
#                 f1_sum += f1

#             avg_precision = precision_sum / num_classes
#             avg_recall = recall_sum / num_classes
#             avg_f1 = f1_sum / num_classes

#             return avg_precision, avg_recall, avg_f1

def Accuracy(outputs, labels):
    _, predictions = torch.max(outputs.data, 1)
    total = 0
    accuracy = 0
    correct = (predictions == labels).sum().item()
    total += labels.size(0)
    accuracy += correct / total
    return accuracy


def Precision(outputs, labels, num_classes=10):
    _, predictions = torch.max(outputs.data, 1)
    precision_sum = 0

    for i in range(num_classes):
        true_positive = ((predictions == i) and (labels == i)).sum().item()
        false_positive = ((predictions == i) and (outputs != i)).sum().item()
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        precision_sum += precision
                
        avg_precision = precision_sum / num_classes
        return avg_precision
          
def Recall(outputs, labels, num_classes=10):
    _, predictions = torch.max(outputs.data, 1)
    recall_sum = 0

    for i in range(num_classes):
        true_positive = ((predictions == i) and (labels == i)).sum().item()
        false_negative = ((predictions != i) and (outputs == i)).sum().item()
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        recall_sum += recall
            
        avg_recall = recall_sum / num_classes
        return avg_recall
      
def F1(outputs, labels):
    precision = Precision(outputs, labels)
    recall = Recall(outputs, labels)

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0   
    return f1
        
