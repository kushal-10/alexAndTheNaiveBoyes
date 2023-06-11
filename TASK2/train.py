from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from helper import tokenizer, data_collator, compute_metrics


def train(n_labels):

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=n_labels)

    if n_labels == 9:
        data = load_from_disk("TASK2/tokens/data1")
        m = "model1"
    elif n_labels == 70:
        data = load_from_disk("TASK2/tokens/data2")
        m = "model2"
    else:
        data = load_from_disk("TASK2/tokens/data3")
        m = "model3"

    training_args = TrainingArguments(
        output_dir="TASK2/"+m,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return None