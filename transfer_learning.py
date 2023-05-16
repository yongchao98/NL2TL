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
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-seed', '--seed', type=int, default=1203) # input random seed
parser.add_argument('-name', '--name', default='GLTL') # The dataset name to train GLTL, circuit, navi
parser.add_argument('-init_weight', '--init_weight', default='with_pre-train') # The initial weight is pre-trained by us or pure from T5
parser.add_argument('-data_size', '--data_size', default='0.1-0.9') # The dataset size range '0.1-0.9' or '0.01-0.09'
parser.add_argument('-model_checkpoint', '--model_checkpoint', default='t5-base') # The model type t5-base and t5-large
args = parser.parse_args()
int_seed = args.seed
dataset_name = args.name
init_weight = args.init_weight
data_size = args.data_size
model_checkpoint = args.model_checkpoint

print(model_checkpoint)
print('*'*20)
print('\n')

# when input data is the ground full NL-TL training dataset
home_path = 'Data_transfer_domain/'
# for word predict
if dataset_name == 'GLTL':
    original_list = [
                     'GLTL_train_8923_for_transfer_word_midfix.jsonl',
                     'GLTL_test_2232_for_transfer_word_midfix.jsonl'
                     ]
elif dataset_name == 'navi':
    original_list = [
                     'navi_total_refined.jsonl'
                     ]
elif dataset_name == 'circuit':
    original_list = [
                     'circuit_total_refined.jsonl'
                     ]
else: print('dataset error!')

dataset_total = []
for file in original_list:
    for line in open(home_path + file, 'r', encoding='utf-8'):  # input data###########################
        dataset_total.append(json.loads(line))
random.shuffle(dataset_total)

import csv
f = open(home_path+'/total_data.csv','w')
csv_writer = csv.writer(f)
csv_writer.writerow(['id',  'ltl', 'sentence'])
for i in range(len(dataset_total)):
  csv_writer.writerow([i, ' '.join(dataset_total[i]['ltl']), ' '.join(dataset_total[i]['sentence'])])
f.close()

dataset = load_dataset('csv', data_files=home_path + '/total_data.csv')

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

if model_checkpoint in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
    prefix = "Transform the following sentence into Signal Temporal logic: "
else:
    prefix = ""

max_input_length = 1024
max_target_length = 128
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

def correct_parenthe(input_str):
  count = 0
  original_list = input_str.split(' ')
  for index, item in enumerate(original_list):
    if len(item) >2:
      if item[-1] == '.':
        original_list[index] = original_list[index][:-1]
    if item == '(':
      count += 1
    elif item == ')':
      count -= 1
  if count >0:
    for i in range(count):
      original_list.append(')')
  if count <0:
    for i in range(-count):
      original_list.pop(-1)
  return ' '.join(original_list)

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
output_dir_total = '../trained_models/' + dataset_name + '/'
if not os.path.exists(output_dir_total):
  os.mkdir(output_dir_total)

if data_size == '0.1-0.9':
    output_dir = output_dir_total+dataset_name+'_varied_dataset_size_seed'+str(int_seed)+'_one'+'_'+init_weight+'_'+model_checkpoint+'/'
elif data_size == '0.01-0.09':
    output_dir = output_dir_total+dataset_name+'_varied_dataset_size_seed'+str(int_seed)+'_pointone'+'_'+init_weight+'_'+model_checkpoint+'/'
if not os.path.exists(output_dir):
  os.mkdir(output_dir)

with open(output_dir +'result.txt', 'w') as f_result:
    for i in range(9):
        if data_size == '0.1-0.9':
            train_dataset, test_dataset = dataset['train'].train_test_split(test_size=0.9-0.1*i).values()
        elif data_size == '0.01-0.09':
            train_dataset, test_dataset = dataset['train'].train_test_split(test_size=0.99 - 0.01 * i).values()
        else: print('Datasize error!')
        tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
        tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

        input_model_dir = 'dir_to_save_the_weights_of_pre-trained_T5_on_lifted_NL_TL/'
        if init_weight == 'with_pre-train':
            model = AutoModelForSeq2SeqLM.from_pretrained(input_model_dir+model_checkpoint+"-finetuned-epoch20/checkpoint-13000")
        elif init_weight == 'without_pre-train':
            model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        else: print('Initial model weights error!')
        batch_size = 16
        model_name = model_checkpoint.split("/")[-1]+'-'+dataset_name+"-epoch20-trainpoint"+str(i+1)
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
            fp16=False,
            #push_to_hub=True,
            #report_to="tensorboard",
            #load_best_model_at_end=True,
            #save_strategy = "no"
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
        #model = AutoModelForSeq2SeqLM.from_pretrained(output_dir+model_name+'/checkpoint-13000').to(device)
        count_correct_w_parenthe = 0
        count_correct_wo_parenthe = 0
        for j in range(min(len(tokenized_test_dataset),1000)):
          inputs = [prefix + tokenized_test_dataset[j]['sentence']]

          inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt").to(device)
          output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=64)
          decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
          #predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]
          predicted_title = decoded_output.strip()
          if correct_parenthe(predicted_title) == tokenized_test_dataset[j]['ltl']:
            count_correct_w_parenthe += 1
          else:
            print(correct_parenthe(predicted_title))
            print(tokenized_test_dataset[j]['ltl'])
            print('\n')
          if predicted_title == tokenized_test_dataset[j]['ltl']:
            count_correct_wo_parenthe += 1
        if data_size == '0.1-0.9':
            print('The training data size is: ', (i+1)*0.1)
            f_result.write(str((i + 1) * 0.1) + '  ' + str(count_correct_w_parenthe / (j+1)) + '  ' + str(count_correct_wo_parenthe / (j+1)) + '\n')
        elif data_size == '0.01-0.09':
            print('The training data size is: ', (i+1) * 0.01)
            f_result.write(str((i + 1) * 0.01) + '  ' + str(count_correct_w_parenthe / (j+1)) + '  ' + str(count_correct_wo_parenthe / (j+1)) + '\n')
        print('The training data number is: ',len(tokenized_train_dataset))
        print('Accuracy with parentheses correction: ', count_correct_w_parenthe/(j+1))
        print('Accuracy without parentheses correction: ', count_correct_wo_parenthe / (j + 1))
f_result.close()
#trainer.save_model()
