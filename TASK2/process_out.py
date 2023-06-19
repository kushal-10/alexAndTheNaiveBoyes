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
all["prediction"]       = [int(e) for e in out ]
all["true_label"]       = [int(e) for e in true_labels]


#select random 20 instances per class that are predicted correctly 
#  -- get lists of indexes - name the lists properly
# print("STUFF")
# print(len(test_df['text'].iloc[40]))
# print(test_df['text'].iloc[40])
#WORK that got predicted correctly:

#Run for every label 
#correct label_predicted label 
event_event = [int(e) for e in list(all[(all['true_label']==2) & (all['prediction']==0)  ]["index_in_dataset"])]

#print(all[["true_label","prediction"]].iloc[18])



model = AutoModelForSequenceClassification.from_pretrained("carbonnnnn/T2L1DISTILBERT", num_labels=9)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False)
#    # outputs a list of dicts like [[{'label': 'NEGATIVE', 'score': 0.0001223755971295759},  {'label': 'POSITIVE', 'score': 0.9998776316642761}]]
#    # print(pipe("I love this movie!"))
#    # model.eval()
# print("PREDICTION from OUT: ")
# print(out[0])
# text = test_df['text'].iloc[5]

# out = pipe(text)  
# print("PREDICTION: ")
# print(int(out[0]['label'][6]))



print(len(event_event))
ut.explain_list(event_event[5::15],"event_agent")
#function explain that takes list of indeces, folder according to list name to save the results

# 9 classes about 10 - 20 samples 
# 9 classes about 10 isclassified samples 

# # draw conclusions
# agent_agent = [int(e) for e in list(all[(all['true_label']==0) & (all['prediction']==0)]["index_in_dataset"])]

# #print(all[["true_label","prediction"]].iloc[18])

# print(len(agent_agent))
# ut.explain_list(agent_agent[:15],"work_work")