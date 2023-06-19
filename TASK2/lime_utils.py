from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from helper import tokenizer, data_collator, compute_metrics
from datasets import load_from_disk
import torch.utils.data as data_utils
import torch.nn as nn
import torch
from datasets import load_dataset
import pickle as pkl
from transformers import AutoTokenizer
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from dill import dump, dumps, loads
import time
from transformers import pipeline
import numpy as np
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
def explain_list(text_ids,file_name):
  class_names = ['Agent',  'Device', 'Event',  'Place',  'Species', 'SportsSeason',  'TopicalConcept',  'UnitOfWork',  'Work']
  explainer = LimeTextExplainer(class_names=class_names)
  id2label = {0: 'Agent', 1: 'Device', 2: 'Event', 3: 'Place', 4: 'Species', 5: 'SportsSeason', 6: 'TopicalConcept', 7: 'UnitOfWork', 8: 'Work'}
  label2id = {'Agent': 0, 'Device': 1, 'Event': 2, 'Place': 3, 'Species': 4, 'SportsSeason': 5, 'TopicalConcept': 6, 'UnitOfWork': 7, 'Work': 8}
  tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
  model = AutoModelForSequenceClassification.from_pretrained("carbonnnnn/T2L1DISTILBERT", num_labels=9, id2label=id2label, label2id=label2id)
  model.eval()
  softmax = nn.Softmax()
  def predictor(texts):
    softmax = nn.Softmax()
    outputs = model(**tokenizer(texts, return_tensors="pt", padding=True))
    probas = softmax(outputs.logits).detach().numpy()
    return probas
  

  dataset = load_dataset("DeveloperOats/DBPedia_Classes")
  test_df = pd.DataFrame(dataset["test"])
  true_labels = torch.load("TASK2/results/true_labels.pt")
  text_ids_all = torch.load("TASK2/results/test_texts.pt")
  out = torch.load("TASK2/results/test_out.pt")

  #EXPLAIN AN INSTANCE:
  explanations = dict()

  for index_in_dataset in text_ids:
    print("Sample "+ str(index_in_dataset))
   #SELECT THE ID OF THE INSTANCE YOU WANT TO EXPLAIN HERE
    
    txt = test_df['text'].iloc[index_in_dataset]
    print(txt)
    ii = list(text_ids_all).index(index_in_dataset)   
    print("INDEX IN list "+str(ii))
    start_time = time.time()
    exp = explainer.explain_instance(txt, predictor, num_features = 10,top_labels=1,num_samples= 800)

    #explanations[index_in_dataset]=exp.as_list(true_labels[index_in_dataset])
    print("--- %s seconds ---" % (time.time() - start_time))

    f = open("TASK2/explainers/"+file_name+"/exp_html_"+str(index_in_dataset)+".html",'w')
 
  
    f.write(exp.as_html(labels = [int(out[ii].item())]))  

  # save dictionary 
  with open("TASK2/results/"+file_name+".pkl", 'wb') as fp:
      pkl.dump(explanations, fp)
      print('dictionary saved successfully to file')

#----------------------------------------------------------------------




with open("TASK2/results/explanations.pkl", 'rb') as fp:
    p = pkl.load(fp)
    #print(p)
   # print('dictionary load successfully from file')

# #GET prediction - so that we would now what label to plot (can plot all if you want)
# classifier = pipeline("text-classification", model="carbonnnnn/T2L1DISTILBERT")
# labeltxt = np.loadtxt("TASK2/label_vals/l1.txt", dtype="str")
# labelint = ['LABEL_0', 'LABEL_1', 'LABEL_2', 'LABEL_3', 'LABEL_4', 'LABEL_5', 'LABEL_6', 'LABEL_7', 'LABEL_8']
# output = classifier(txt)[0]['label']
# label_i= int(output[6]) #to get the index
# #-----------------------------------------------------------------------------------




