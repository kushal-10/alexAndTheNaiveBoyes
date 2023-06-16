from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TextClassificationPipeline, AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import torch 

class_names = ['Agent',  'Device', 'Event',  'Place',  'Species', 'SportsSeason',  'TopicalConcept',  'UnitOfWork',  'Work']
label_dict=dict(zip(class_names,range(9)))

# LOAD TEST DATASET
dataset = load_dataset("DeveloperOats/DBPedia_Classes")
test_df = pd.DataFrame(dataset["test"])

all = pd.DataFrame()

text_ids = torch.load("TASK2/results/test_texts.pt")
out = torch.load("TASK2/results/test_out.pt")
true_labels = torch.load("TASK2/results/true_labels.pt")

true_labels= []

labelstr = np.loadtxt('TASK2/label_vals/l1.txt', dtype="str")
for i in range(len(text_ids)):
    s=test_df['l1'].iloc[i]
    true_labels.append(label_dict[s])
torch.save(true_labels, "TASK2/results/true_labels.pt")

true_labels = torch.load("TASK2/results/true_labels.pt")
print(len(text_ids))
print(len(out))
print(true_labels[14])
