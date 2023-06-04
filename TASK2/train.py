from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from dataset2 import tokenizer, data_collator, compute_metrics

id2label = {0: 'Agent', 1: 'Device', 2: 'Event', 3: 'Place', 4: 'Species', 5: 'SportsSeason', 6: 'TopicalConcept', 7: 'UnitOfWork', 8: 'Work'}

label2id = {'Agent': 0, 'Device': 1, 'Event': 2, 'Place': 3, 'Species': 4, 'SportsSeason': 5, 'TopicalConcept': 6, 'UnitOfWork': 7, 'Work': 8}

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=9, id2label=id2label, label2id=label2id)

data1 = load_from_disk("TASK2/tokens/data1")
data1 = data1.rename_column("label", "labels")
data1 = data1.with_format("torch")
# data1 = data1.remove_columns(data1["train"].column_names)
# print(data1["test"][0])

training_args = TrainingArguments(
    output_dir="TASK2/model1",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data1["train"],
    eval_dataset=data1["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
# print(trainer.train_dataset[0])