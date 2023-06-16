from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TextClassificationPipeline, AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import torch 
import lime_utils as ut

class_names = ['Agent',  'Device', 'Event',  'Place',  'Species', 'SportsSeason',  'TopicalConcept',  'UnitOfWork',  'Work']
label_dict=dict(zip(class_names,range(9)))

# LOAD TEST DATASET
dataset = load_dataset("DeveloperOats/DBPedia_Classes")
test_df = pd.DataFrame(dataset["test"])

all = pd.DataFrame()

text_ids = torch.load("TASK2/results/test_texts.pt")
out = torch.load("TASK2/results/test_out.pt")
true_labels = torch.load("TASK2/results/true_labels.pt")

out= [int(it.item()) for it in out]

# print(txt)
# print(true_labels[i])
# print(out[i])

all["index_in_dataset"] = text_ids
all["text"]             = test_df['text']
all["prediction"]       = out
all["true_label"]       = true_labels


#select random 20 instances per class that are predicted correctly 
#  -- get lists of indexes - name the lists properly

#WORK that got predicted correctly:
#correct label_predicted label 
work_work = [int(e) for e in list(all[(all['true_label']==8) & (all['prediction']==8)]["index_in_dataset"])]



#try out a pass 


model = AutoModelForSequenceClassification.from_pretrained("carbonnnnn/T2L1DISTILBERT", num_labels=9)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False)
print(len(work_work))
ut.explain_list(work_work,"work_work")
#function explain that takes list of indeces, folder according to list name to save the results