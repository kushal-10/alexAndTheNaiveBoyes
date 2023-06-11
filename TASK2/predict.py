from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TextClassificationPipeline, AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt


def predictions():

   model = AutoModelForSequenceClassification.from_pretrained("TASK2/model1/checkpoint-60236", num_labels=9)
   tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

   pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False)
   # outputs a list of dicts like [[{'label': 'NEGATIVE', 'score': 0.0001223755971295759},  {'label': 'POSITIVE', 'score': 0.9998776316642761}]]
   # print(pipe("I love this movie!"))
   # model.eval()
   
   # LOAD TEST DATASET
   dataset = load_dataset("DeveloperOats/DBPedia_Classes")
   test_df = pd.DataFrame(dataset["test"])

   predictions = []
   skip = []
   for i in range(len(test_df)):
      if i%1000 == 0:
         print("Evaluation for " + str(i) + " samples Done.....")

      text = test_df['text'].iloc[i]

      # SKIP TEXTS HAVING LENGTH LONGER THAN 1000 
      if len(text) > 1000:
         skip.append(str(i))
      else:
         out = pipe(text)
         predictions.append(out[0]['label'])

   np.savetxt('TASK2/results/predictions.txt', predictions, fmt="%s")
   np.savetxt('TASK2/results/skip.txt', skip, fmt="%s")
   save_cfm()

   return None

def save_cfm():
   dataset = load_dataset("DeveloperOats/DBPedia_Classes")
   test_df = pd.DataFrame(dataset["test"])

   # prepare y
   y_temp = []
   for i in range(len(test_df)):
      y_temp.append(test_df['l1'].iloc[i])

   skips = np.loadtxt('TASK2/results/skip.txt', dtype="str")
   skip_ints = []
   for i in range(len(skips)):
      skip_ints.append(int(skips[i]))
   skip_ints.reverse()

   for i in range(len(skip_ints)):
      y_temp.pop(skip_ints[i])

   # PREPARE y_preds
   labelnum = ['LABEL_0', 'LABEL_1', 'LABEL_2', 'LABEL_3', 'LABEL_4', 'LABEL_5', 'LABEL_6', 'LABEL_7', 'LABEL_8']
   labelstr = np.loadtxt('TASK2/label_vals/l1.txt', dtype="str")
   labeldict = {}
   for l in range(len(labelnum)):
      labeldict[labelnum[l]] = labelstr[l]

   preds = np.loadtxt('TASK2/results/predictions.txt', dtype="str")
   y_preds = []
   for p in range(len(preds)):
      s = preds[p]
      y_preds.append(labeldict[s])

   # GET CFM
   cfm = confusion_matrix(y_temp, y_preds, labels=labelstr)
   classes = labelstr

   df_cfm = pd.DataFrame(cfm, index = classes, columns = classes)
   plt.figure(figsize = (20,15))
   cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues')
   cfm_plot.figure.savefig("TASK2/figures/cfm.png")
   df_cfm.to_csv("TASK2/tables/cfm.csv")


   return None


save_cfm()