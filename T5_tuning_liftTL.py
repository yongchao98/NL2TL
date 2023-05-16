import nltk
nltk.download('punkt')
import transformers
model_checkpoint = "t5-large" # can choose t5-large or t5-base
print(model_checkpoint)
print('*'*20)
print('\n')

# Directly from json file to the dataset_total
from IPython.core import error
import json
from fnmatch import fnmatchcase as match
import random
import os
import pandas as pd
import datasets
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
import nltk
import numpy as np

# when input data is the real training dataset
home_path = 'supple_data/Data_word_inorder_21867/'
# for word predict
original_list = ['combine_dev_seq2tree_idea4.jsonl',
                 'combine_train_seq2tree_idea4.jsonl',
                 'combine_test_seq2tree_idea4.jsonl'
                 ]

dataset_total = []
for file in original_list:
    for line in open(home_path + file, 'r', encoding='utf-8'):  # input data###########################
        dataset_total.append(json.loads(line))
random.shuffle(dataset_total)

len_train = int(0.7*len(dataset_total)); len_dev = int(0.85*len(dataset_total));

import csv
f = open(home_path+'/total_data.csv','w')
csv_writer = csv.writer(f)
csv_writer.writerow(['id', 'logic_ltl', 'logic_sentence', 'ltl', 'sentence'])
for i in range(len(dataset_total)):
  csv_writer.writerow([i, ' '.join(dataset_total[i]['logic_ltl']), ' '.join(dataset_total[i]['logic_sentence']), ' '.join(dataset_total[i]['ltl']), ' '.join(dataset_total[i]['sentence'])])
f.close()

dataset = load_dataset('csv', data_files=home_path + '/total_data.csv')

train_dataset, test_dataset= dataset['train'].train_test_split(test_size=0.1).values()
#dev_dataset, test_dataset = validation_dataset.train_test_split(test_size=0.5).values()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

if model_checkpoint in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
    prefix = "Transform the following sentence into Signal Temporal logic: "
else:
    prefix = ""

max_input_length = 1024
max_target_length = 128
def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["logic_sentence"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["logic_ltl"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["logic_sentence"] = examples["logic_sentence"]
    model_inputs["logic_ltl"] = examples["logic_ltl"]
    model_inputs["id"] = examples["id"]
    return model_inputs

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)
#tokenized_dev_dataset = dev_dataset.map(preprocess_function, batched=True)

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
batch_size = 16
model_name = model_checkpoint.split("/")[-1]+"-epoch20-infix-word"
output_dir = '../trained_models/'
model_dir = output_dir+model_name

args = Seq2SeqTrainingArguments(
    model_dir,
    model_name,
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    seed=1203,
    save_total_limit=1,
    num_train_epochs=20,
    predict_with_generate=True,
    fp16=False,
    #push_to_hub=True,
    #report_to="tensorboard",
    #load_best_model_at_end=True,
    #save_strategy = "no"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # print(predictions)
    # print(labels)
    # Replace -100 in the labels as we can't decode them.
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    count = 0
    for i in range(len(decoded_preds)):
        pred = nltk.sent_tokenize(decoded_preds[i].strip())
        label = nltk.sent_tokenize(decoded_labels[i].strip())
        if pred == label:
            count += 1
    return {'top-1 accuracy': round(count / len(decoded_preds), 6)}

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_train_dataset ,
    eval_dataset=tokenized_test_dataset ,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()

import torch
from transformers import AutoModelForSeq2SeqLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
count_correct = 0
for i in range(min(len(tokenized_test_dataset),1000)):
  inputs = [prefix + tokenized_test_dataset[i]['logic_sentence']]
  inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt").to(device)
  output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=64)
  decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
  predicted_title = decoded_output.strip()
  if predicted_title == tokenized_test_dataset[i]['logic_ltl']:
    count_correct += 1
  else:
    print(predicted_title)
    print(tokenized_test_dataset[i]['logic_ltl'])
    print('\n')
print('Accuracy: ', count_correct/(i + 1))
print('\n'*2)

with open(output_dir +'result_'+model_name+'.txt', 'w') as f_result:
 f_result.write(model_name+' accuracy is:  ' + str(count_correct / (i + 1)) + '\n')
f_result.close()
