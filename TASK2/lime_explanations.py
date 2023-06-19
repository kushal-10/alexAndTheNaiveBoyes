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
text_ids_all = torch.load("TASK2/results/test_texts.pt")
labelstr = np.loadtxt('TASK2/label_vals/l1.txt', dtype="str")
for i in text_ids:
   
    s=test_df['l1'].iloc[int(i)]
    true_labels.append(label_dict[s])
torch.save(true_labels, "TASK2/results/true_labels.pt")

true_labels = torch.load("TASK2/results/true_labels.pt")
print(len(text_ids))
print(len(out))
print(test_df['l1'].iloc[56078])

index_in_dataset = 56078 #SELECT THE ID OF THE INSTANCE YOU WANT TO EXPLAIN HERE
txt = test_df["text"][index_in_dataset]
print(txt)