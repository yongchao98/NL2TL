import nltk
nltk.download('punkt')
import transformers

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
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-seed', '--seed', type=int, default=1203) # input random seed
parser.add_argument('-init_weight', '--init_weight', default='with_pre-train') # The initial weight is pre-trained by us or pure from T5
parser.add_argument('-model_checkpoint', '--model_checkpoint', default='t5-base') # The model type t5-base and t5-large
args = parser.parse_args()

int_seed = args.seed
init_weight = args.init_weight
model_checkpoint = args.model_checkpoint

print(model_checkpoint)
print('*'*20)
print('\n')

# when input data is the real training dataset
home_path = 'Data_transfer_domain/'
# for word predict
original_list = [
                 'CW_total_3382_for_transfer_word_midfix.jsonl'
                 ]

dataset_total = []
for file in original_list:
    for line in open(home_path + file, 'r', encoding='utf-8'):  # input data###########################
        dataset_total.append(json.loads(line))
random.shuffle(dataset_total)

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["sentence"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["ltl"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["sentence"] = examples["sentence"]
    model_inputs["ltl"] = examples["ltl"]
    model_inputs["id"] = examples["id"]
    return model_inputs

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

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

if model_checkpoint in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
    prefix = "Transform the following sentence into Signal Temporal logic: "
else:
    prefix = ""

max_input_length = 1024
max_target_length = 128

unique_ltl = []
for i in range(len(dataset_total)):
  if dataset_total[i]['ltl'] not in unique_ltl:
    unique_ltl.append(dataset_total[i]['ltl'])

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

input_model_dir = 'dir_to_save_the_weights_of_pre-trained_T5_on_lifted_NL_TL/'
if init_weight == 'with_pre-train':
    model = AutoModelForSeq2SeqLM.from_pretrained(
        input_model_dir + model_checkpoint + "-finetuned-epoch20/checkpoint-13000")
elif init_weight == 'without_pre-train':
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
else:
    print('Initial model weights error!')
batch_size = 16

dataset_name = 'CW'
output_dir_total = '../trained_models/' + dataset_name + '/'
if not os.path.exists(output_dir_total):
  os.mkdir(output_dir_total)

output_dir = output_dir_total+dataset_name+'_varied_dataset_size_seed'+str(int_seed)+'_'+init_weight+'_'+model_checkpoint+'/'
if not os.path.exists(output_dir):
  os.mkdir(output_dir)

with open(output_dir +'result.txt', 'w') as f_result:
    for a in range(4,40,4):
        random.shuffle(unique_ltl)
        dataset_train = []; dataset_test = []
        for i in range(len(dataset_total)):
            if dataset_total[i]['ltl'] in unique_ltl[0:a]:
              dataset_train.append(dataset_total[i])
            else:
              dataset_test.append(dataset_total[i])
        print('The num of training class types is: ', a)
        print('The num of testing class types is: ', 39-a)
        print('The num of training dataset is: ', len(dataset_train))
        print('The num of testing dataset is: ',len(dataset_test))
        print('/n'*2)

        import csv
        f = open(home_path+'/train_data.csv','w')
        csv_writer = csv.writer(f)
        csv_writer.writerow(['id', 'ltl', 'sentence'])
        for i in range(len(dataset_train)):
          csv_writer.writerow([i, ' '.join(dataset_train[i]['ltl']), ' '.join(dataset_train[i]['sentence'])])
        f.close()

        f = open(home_path+'/test_data.csv','w')
        csv_writer = csv.writer(f)
        csv_writer.writerow(['id', 'ltl', 'sentence'])
        for i in range(len(dataset_test)):
          csv_writer.writerow([i, ' '.join(dataset_test[i]['ltl']), ' '.join(dataset_test[i]['sentence'])])
        f.close()

        data_files = {"train": home_path + '/train_data.csv', "test": home_path + '/test_data.csv'}
        dataset = load_dataset("csv", data_files=data_files)

        train_dataset, test_dataset= dataset.values()

        tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
        tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

        model_name = model_checkpoint.split("/")[-1]+"-CW-epoch20-"+'train_typenum'+str(a)
        model_dir = output_dir+model_name

        args = Seq2SeqTrainingArguments(
            model_dir,
            model_name,
            evaluation_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            seed=int_seed,
            save_total_limit=1,
            num_train_epochs=20,
            predict_with_generate=True,
            fp16=False
        )
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

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
          inputs = [prefix + tokenized_test_dataset[i]['sentence']]
          inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt").to(device)
          output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=64)
          decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
          predicted_title = decoded_output.strip()
          if predicted_title == tokenized_test_dataset[i]['ltl']:
            count_correct += 1
          else:
            print(predicted_title)
            print(tokenized_test_dataset[i]['ltl'])
            print('\n')
        print('The training  type number is: ', a)
        print('Accuracy: ', count_correct/(i + 1))
        print('\n'*2)
        f_result.write(str(a) + '  ' + str(count_correct / (i + 1)) + '\n')
f_result.close()
#trainer.save_model()
