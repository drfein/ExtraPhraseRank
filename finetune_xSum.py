# Import
import torch
import numpy as np
import datasets

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

from tabulate import tabulate
import nltk
from datetime import datetime
import wandb

wandb.login()

language = "english"
torch.manual_seed(0)
# Get model and tokenizer
model_name = "facebook/bart-base"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set length parameters
encoder_max_length = 256  # demo
decoder_max_length = 64

# Download dataset
data = datasets.load_dataset("xsum", split='train[:1000]')

# Format and split data
def flatten(example):
    return {
        "document": example["document"],
        "summary": example["summary"],
    }

def list2samples(example):
    documents = []
    summaries = []
    for doc, sum in zip(example["document"], example["summary"]):
        if len(doc) > 0:
            documents.append(doc)
            summaries.append(sum)
    return {"document": documents, "summary": summaries}


dataset = data.map(flatten, remove_columns=["id"])
dataset = dataset.map(list2samples, batched=True)

train_data_txt, validation_data_txt = dataset.train_test_split(test_size=0.1).values()


# Tokenize data
def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length):
    source, target = batch["document"], batch["summary"]
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    batch = {k: v for k, v in source_tokenized.items()}
    # Ignore padding in the loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch


train_data = train_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=train_data_txt.column_names,
)

validation_data = validation_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=validation_data_txt.column_names,
)


nltk.download("punkt", quiet=True)
metric = datasets.load_metric("rouge")
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# Configure training

training_args = Seq2SeqTrainingArguments(
    output_dir="results",
    num_train_epochs=1,  # demo
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=4,  # demo
    per_device_eval_batch_size=4,
    # learning_rate=3e-05,
    warmup_steps=500,
    weight_decay=0.1,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
    logging_dir="logs",
    logging_steps=50,
    save_total_limit=3,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=validation_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# W and B logging
wandb_run = wandb.init(
        project="bart_for_sum",
        config={
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "learning_rate": training_args.learning_rate,
            "dataset": "wiki_lingua " + language,
        },
    )

now = datetime.now()
current_time = now.strftime("%H%M%S")
wandb_run.name = "run_" + language + "_" + current_time

# Evaluate the model before and after training
trainer.evaluate()
trainer.train()
trainer.evaluate()

# Stop logging
wandb_run.finish()