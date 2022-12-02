import os
import json
import time
from argparse import ArgumentParser
import tqdm
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from transformers import T5Tokenizer, AdamW, get_linear_schedule_with_warmup
from seq2seq import LTL2Eng
from data import LTLDataset, generate_vocabs, LTL2EngT5Dataset, UnlabeledLTLDataset
from config import Config
from util import collect_ltl_vocabs
import pandas as pd
home_path_nl2stl = '.'
config = Config.from_json_file('eng2ltl_t5_load_data.json')
input_dir = os.path.join(home_path_nl2stl, 'eng2ltl_weights_11_28_word_infix')

torch.cuda.set_device(0)
model_name = config.bert_model_name
tokenizer = T5Tokenizer.from_pretrained(model_name,cache_dir=config.bert_cache_dir)

ltl_vocabs = collect_ltl_vocabs([home_path_nl2stl+config.train_file,home_path_nl2stl+config.dev_file,home_path_nl2stl+config.test_file])
model = LTL2Eng(config,ltl_vocabs,tokenizer)
model.load_state_dict(torch.load(input_dir+'/model_state.pt'))
model.cuda(device=0)
model.eval()

# Here you can define the imperative natural sentences to express orders, like the following example
# Here we represent atomic propositions with ( prop_* )

test_sentence = [
    'If ( prop_4 ) happens and implies ( prop_2 ) and this scenario continues to hold until at some point during the 243 to 582 time units ( prop_3 ) is detected , then the scenario is equivalent to ( prop_1 ) .',
    'If it is not the case that ( prop_3 ) is detected for each time instant in the coming 164 to 612 time units , or else ( prop_1 ) happens , then ( prop_2 ) .',
    'If at some point ( prop_3 ) and ( prop_2 ) , and is equivalent to ( prop_4 ) , and this scenario will hold until at some other point ( prop_1 ) is detected .',

    ]

'''
# Another way is to load the excel files, there are some example files in the dataset dir
file_name = '/output_davinci_gen_ltl_long_8'
df = pd.read_excel(home_path_nl2stl+'/dataset/'+file_name+'.xlsx')
test_sentence = [df['paraphrased_logic_sentence'][i] for i in range(len(df))]
'''

home_path_output = home_path_nl2stl + '/application_test/'
if not os.path.exists(home_path_output):
  os.mkdir(home_path_output)

dataset_total = [];
with open(home_path_output+"test1_word_infix.jsonl", "w") as outfile:
    for i in range(len(test_sentence)):
      dataset_item = {};
      dataset_item['id'] = i
      dataset_item['logic_ltl'] = ''
      dataset_item['logic_sentence'] = test_sentence[i].split(' ')
      outfile.write(json.dumps(dataset_item)+'\n')
outfile.close()

test_set = LTL2EngT5Dataset(home_path_output+"test1_word_infix.jsonl",tokenizer,config,model.ltl2eng.config,ltl_vocabs=ltl_vocabs)
data_batch =  DataLoader(test_set, batch_size=1,
                        shuffle=False, collate_fn=test_set.collate_fn)

for i, batch in enumerate(data_batch):
  if i <len(test_sentence):
    outputs = model.predict_eng2ltl(batch, max_length=config.max_generate_length)
    pred_ltls = tokenizer.batch_decode(outputs['output_idxs'], skip_special_tokens=True)
    print(test_sentence[i])
    print(pred_ltls)
    print('\n')
  else:
    break
