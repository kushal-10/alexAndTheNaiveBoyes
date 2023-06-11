from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from helper import tokenizer, data_collator, compute_metrics
from datasets import load_from_disk
import torch.utils.data as data_utils
import torch.nn as nn
import torch

from transformers import AutoTokenizer
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from dill import dump, dumps, loads

class_names = ['Agent',  'Device', 'Event',  'Place',  'Species', 'SportsSeason',  'TopicalConcept',  'UnitOfWork',  'Work']

explainer = LimeTextExplainer(class_names=class_names)

id2label = {0: 'Agent', 1: 'Device', 2: 'Event', 3: 'Place', 4: 'Species', 5: 'SportsSeason', 6: 'TopicalConcept', 7: 'UnitOfWork', 8: 'Work'}
label2id = {'Agent': 0, 'Device': 1, 'Event': 2, 'Place': 3, 'Species': 4, 'SportsSeason': 5, 'TopicalConcept': 6, 'UnitOfWork': 7, 'Work': 8}
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#Load model:
model = AutoModelForSequenceClassification.from_pretrained("carbonnnnn/T2L1DISTILBERT", num_labels=9, id2label=id2label, label2id=label2id)
model.eval()


#LOAD DATA
data1 = load_from_disk("TASK2/tokens/data1")
data1 = data1.with_format("torch")


def predictor(texts):
  softmax = nn.Softmax()
  outputs = model(**tokenizer(texts, return_tensors="pt", padding=True))
  probas = softmax(outputs.logits).detach().numpy()
  return probas


#EXPLAIN AN INSTANCE:

index_in_dataset = 200  #SELECT THE ID OF THE INSTANCE YOU WANT TO EXPLAIN HERE
txt = data1["test"][index_in_dataset]["text"]


exp = explainer.explain_instance(txt, predictor, num_features = 10,labels = (0,1,2,3,4,5,6,7,8),num_samples= 100 )

#SAVE EXPALINER OBJECT SO THAT IT CAN BE LOADED LATER -- gives errors, will figure out later 
# name = "TASK2/explainers/"+str(index_in_dataset)
# f = open(name,'w')
# f.write(dumps(exp))

#GET prediction - so that we would now what label to plot (can plot all if you want)
from transformers import pipeline
import numpy as np
classifier = pipeline("text-classification", model="carbonnnnn/T2L1DISTILBERT")
labeltxt = np.loadtxt("TASK2/label_vals/l1.txt", dtype="str")
labelint = ['LABEL_0', 'LABEL_1', 'LABEL_2', 'LABEL_3', 'LABEL_4', 'LABEL_5', 'LABEL_6', 'LABEL_7', 'LABEL_8']
output = classifier(txt)[0]['label']
label_i= int(output[6]) #to get the index


f = open("TASK2/explainers/exp_html_"+str(index_in_dataset)+".html",'w')
f.write(exp.as_html(labels = [label_i]))  
#predict_proba(txt)
print(exp.as_list())