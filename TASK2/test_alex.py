from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from dataset2 import tokenizer, data_collator, compute_metrics
from datasets import load_from_disk
import torch.utils.data as data_utils
import torch.nn as nn
import torch
from lime import lime_text
from lime.lime_text import LimeTextExplainer

class_names = ['Agent',  'Device', 'Event',  'Place',  'Species', 'SportsSeason',  'TopicalConcept',  'UnitOfWork',  'Work']



explainer = LimeTextExplainer(class_names=class_names)


id2label = {0: 'Agent', 1: 'Device', 2: 'Event', 3: 'Place', 4: 'Species', 5: 'SportsSeason', 6: 'TopicalConcept', 7: 'UnitOfWork', 8: 'Work'}

label2id = {'Agent': 0, 'Device': 1, 'Event': 2, 'Place': 3, 'Species': 4, 'SportsSeason': 5, 'TopicalConcept': 6, 'UnitOfWork': 7, 'Work': 8}

#Load model:
model = AutoModelForSequenceClassification.from_pretrained("TASK2/model1/checkpoint-500", num_labels=9, id2label=id2label, label2id=label2id)
model.eval()

data1 = load_from_disk("TASK2/tokens/data1")
d1 = load_from_disk("TASK2/tokens/data1")
data1 = data1.rename_column("label", "labels")
data1 = data1.with_format("torch")


print(data1["test"][28]['text'])
t = data1["test"][28]["input_ids"].view(1,-1)

txt = data1["test"][500]["text"]

p = model(t)['logits']

softmax = nn.Softmax()
print(softmax(p))

#Get one instance from test: 


def predict_prob(texts):
    preds =[]
    for text in texts:
        t = tokenizer(text)
       # print(t["input_ids"].type())
  
        p = model(torch.LongTensor(t["input_ids"]).view(1,-1))['logits']
        softmax = nn.Softmax()
        i=softmax(p)
        preds.append(i)
    return preds

def predictor(texts):
  outputs = model(**tokenizer(texts, return_tensors="pt", padding=True))
  probas = softmax(outputs.logits).detach().numpy()
  return probas


from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from sklearn.pipeline import make_pipeline
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


exp = explainer.explain_instance(txt, predictor, num_features = 10, num_samples= 2000)

f = open('exp.html','w')
f.write(exp.as_html())
#predict_proba(txt)
print(exp.as_list())