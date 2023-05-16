# Generate the STL-NL pairs
import copy
import random
from fnmatch import fnmatchcase as match
import numpy as np
import json
import os
import openai
from tqdm import tqdm
import csv
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-n', '--number', type=int, default=10)
parser.add_argument('-f', '--filename', type=int, default=0)
args = parser.parse_args()
init_num = args.number
filename = args.filename

time_upper_limit = 600

operation_list = ['negation', '&', '->', '<->', '|', 'G', 'F', 'U', 'G[*,**', 'F[*,**', 'U[*,**']
two_leaves = ['&', '->', '<->', '|', 'U', 'U[*,**']
one_leaf = ['negation', 'G', 'F', 'G[*,**', 'F[*,**']
no_leaf = ['prop**']
prop_with_time = ['G[*,**', 'F[*,**', 'U[*,**']

def operation2word(pre_order_list):
  dict_operation = {}
  dict_operation['&'] = 'and'; dict_operation['->'] = 'imply'; dict_operation['<->'] = 'equal'; dict_operation['|'] = 'or'
  dict_operation['G'] = 'globally'; dict_operation['F'] = 'finally'; dict_operation['U'] = 'until'; dict_operation['negation'] = 'negation'
  post_list = []
  for item in pre_order_list:
    if match(item, 'G[*,**'):
      post_list.append('globally'+item[1:])
    elif match(item, 'F[*,**'):
      post_list.append('finally'+item[1:])
    elif match(item, 'U[*,**'):
      post_list.append('until'+item[1:])
    elif match(item, 'prop**'):
      post_list.append(item)
    else:
      post_list.append(dict_operation[item])
  return post_list

def judge_leaf_num(item):
  two_leaves = ['&', '->', '<->', '|', 'U', 'U[*,**']
  one_leaf = ['negation', 'G', 'F', 'G[*,**', 'F[*,**']
  if item in two_leaves or item[0]=='U':
    return 2
  elif match(item, 'prop**'):
    return 0
  else:
    return 1

def pre_to_mid_exp(item_list_original):
  item_list = copy.deepcopy(item_list_original)
  item_list.reverse()
  word_list = operation2word(item_list)
  stack_list = []; stack_list_2 = []
  mark = 0
  for i,item in enumerate(item_list):
    if judge_leaf_num(item) == 0:
      stack_list.append(word_list[i])
      stack_list_2.append(item_list[i])
    elif judge_leaf_num(item) == 1:
      if mark:
        candidate = word_list[i] + ' '+ '(' + stack_list[-1] + ')'
        candidate_2 = item_list[i] + ' '+ '(' + stack_list_2[-1] + ')'
      else:
        candidate = word_list[i] + ' '+ stack_list[-1]
        candidate_2 = item_list[i] + ' '+ stack_list_2[-1]
      stack_list.pop(-1)
      stack_list.append(candidate)
      stack_list_2.pop(-1)
      stack_list_2.append(candidate_2)
      mark = 1
    elif judge_leaf_num(item) == 2:
      candidate = '('+ stack_list[-1] + ' ' + word_list[i] + ' ' + stack_list[-2] + ')'
      candidate_2 = '('+ stack_list_2[-1] + ' ' + item_list[i] + ' ' + stack_list_2[-2] + ')'
      stack_list.pop(-1)
      stack_list.pop(-1)
      stack_list.append(candidate)
      stack_list_2.pop(-1)
      stack_list_2.pop(-1)
      stack_list_2.append(candidate_2)
      mark=0
  return candidate, candidate_2

def generate_ltl_from_list(raw_list):
  base_list = copy.deepcopy(raw_list)
  if len(base_list)==1:
    random_num_2 = random.randint(0, len(one_leaf)-1)
    if one_leaf[random_num_2] in prop_with_time:
      time1 = random.randint(1, 500); time2 = random.randint(1, 500) + time1
      if time2 > time_upper_limit:
        base_list.insert(0, one_leaf[random_num_2][0:2]+str(time1)+','+'infinite'+']')
      else:
        base_list.insert(0, one_leaf[random_num_2][0:2]+str(time1)+','+str(time2)+']')
    else:
      base_list.insert(0, one_leaf[random_num_2])
  elif len(base_list)>1:
    count = len(base_list)
    while count>1:
      random_num_3 = random.randint(0, len(operation_list)-1)
      if operation_list[random_num_3] in prop_with_time:
        time1 = random.randint(1, 500); time2 = random.randint(1, 500) + time1
        base_list.insert(0, operation_list[random_num_3][0:2]+str(time1)+','+str(time2)+']')
      else:
        base_list.insert(0, operation_list[random_num_3])
      count -= judge_leaf_num(operation_list[random_num_3])
      count += 1
  return base_list

def generate_ltl_v2(prop_num = 6):
  random_num = random.randint(2, prop_num)
  num_prop = [i for i in range(1,random_num)]
  random.shuffle(num_prop)
  father_list = ['prop_'+str(item) for item in num_prop]
  if len(father_list) > 1:
    divide_index = random.randint(1, len(father_list)-1)
    son_list1 = generate_ltl_from_list(father_list[0 : divide_index])
    son_list2 = generate_ltl_from_list(father_list[divide_index : len(father_list)])
    random_num_1 = random.randint(0, len(two_leaves)-1)

    mid_two_leaves_item = []
    if two_leaves[random_num_1] in prop_with_time:
      time1 = random.randint(1, 500); time2 = random.randint(1, 500) + time1
      if time2 > time_upper_limit:
        mid_two_leaves_item.append(two_leaves[random_num_1][0:2]+str(time1)+','+'infinite'+']')
      else:
        mid_two_leaves_item.append(two_leaves[random_num_1][0:2]+str(time1)+','+str(time2)+']')
    else:
      mid_two_leaves_item.append(two_leaves[random_num_1])

    total_list = mid_two_leaves_item + son_list1 + son_list2
    return total_list

# generate STLs
list_ltl = []
print('-'*40)
print('Generating rule based STL: ')
while len(list_ltl) < init_num:
  ltl_can = generate_ltl_v2(9)
  mark = 1
  if ltl_can is not None and len(ltl_can) <8 and len(ltl_can) >5:
  #if len(ltl_can) <=5 and len(ltl_can) >=3:
    count_time = 0; count_FGU = 0
    count_negation = 0; count_u = 0
    for i,item in enumerate(ltl_can):
      if item[0] == 'U':
        count_u += 1
      if match(item, 'G[*,**') or match(item, 'F[*,**') or match(item, 'U[*,**'):
        if i<len(ltl_can)-1:
          if item[0] == ltl_can[i+1][0]:
            mark = 0
        count_time += 1
      elif item == 'negation':
        count_negation += 1
      elif item in ['U', 'F', 'G']:
        count_FGU += 1
      if item in ['U', 'F', 'G', '<->']:
        if i<len(ltl_can)-1:
          if item == ltl_can[i+1]:
            mark = 0
          if item in ['U', 'F', 'G'] and item[0] == ltl_can[i+1][0]:
            mark = 0
    if (count_FGU + count_time)<2 and count_negation<2 and mark and not ltl_can in list_ltl and count_u <2:
      list_ltl.append(ltl_can)
      #print(ltl_can)
      word_express, operation_express = pre_to_mid_exp(ltl_can)
      #print(word_express)
      #print(operation_express)
      #print('\n')

# Generate NL-1 from STL-1
def paraphrase_GPT3(original_sent):
  openai.api_key = '' # input your openai key
  response = openai.Completion.create(
    model="text-davinci-002",

    # Prompt for predicting NL from LTL
    #prompt="Try to transform the signal temporal logic into natural languages, the operators in the signal temporal logic are: negation, imply, and, equal, until, globally, finally, or .\nThe examples are as following:\n\nLTL: (finally [317,767] (prop_3 equal prop_2) imply prop_1)\nnatural language: It is required that at a certain point within the next 317 to 767 time units the scenario in which ( prop_3 ) is equivalent to the scenario in which ( prop_2 ) happens , and only then ( prop_1 ) .\n\nLTL: (((prop_4 or prop_2) or prop_3) until prop_1)\nnatural language: In case that at some point ( prop_4 ) or ( prop_2 ) or ( prop_3 ) is detected and continued until then at some other point ( prop_1 ) should be detected as well .\n\nLTL: (((prop_2 until [417,741] prop_3) imply prop_1) or prop_4)\nnatural language: ( prop_2 ) should happen and hold until at a certain time point during the 417 to 741 time units the scenario that ( prop_3 ) should happen then ( prop_1 ) , or else ( prop_4 ) .\n\nLTL: ((prop_3 until [500,903] prop_1) and negation prop_2)\nnatural language:  ( prop_3 ) happens and continues until at some point during the 500 to 903 time units ( prop_1 ) happens , and in the same time ( prop_2 ) does not happen .\n\nLTL: (globally [107,513] prop_1 or (prop_3 and prop_2))\nnatural language: For each time instant in the next 107 to 513 time units ( prop_1 ) is true , or else ( prop_3 ) happens and ( prop_2 ) happens at the same time.\n\nLTL: ((prop_3 and prop_2) or (prop_4 until[469,961] prop_1))\nnatural language:  ( prop_3 ) and ( prop_2 ) should happen at the same time , or else ( prop_4 ) happens and continues until at some point during the 469 to 961 time units ( prop_1 ) happens .\n\nLTL: (negation prop_2 or (prop_1 until[286,348] prop_3))\nnatural language:  ( prop_2 ) should not happen , or else ( prop_1 ) happens and continues until at some point during the 286 to 348 time units ( prop_3 ) happens .\n\nLTL: ((prop_1 imply prop_2) and (prop_3 imply negation prop_2))\nnatural language: While (prop_1) , do (prop_2) , and when (prop_3) , stop (prop_2) .\n\nLTL: (prop_1 imply finally [300, infinite] prop_2)\nnatural language: If (prop_1) happens, then some time after the next 300 time steps (prop_2) should happen.\n\nLTL: (prop_1 imply (globally prop_2 and (prop_3 imply finally [0, 10] prop_4)))\nnatural language: If (prop_1) happens, then for all time afterward (prop_2) holds and if, in addition, if (prop_3) occurs, then (prop_4) eventually occurs in the next 10 time units.\n\nLTL: (prop_1 imply (negation prop_2 and (prop_3 until prop_4)))\nnatural language: If (prop_1), don't (prop_2), instead keep (prop_3) until (prop_4).\n\nLTL: (prop_4 imply ((prop_1 or prop_2) or prop_3))\nnatural language: If (prop_4), then make sure any of the following happens: (prop_1), (prop_2) or (prop_3).\n\nLTL: (finally [0, 500] (prop_2) imply finally [0, 999] prop_1)\nnatural language: Always make (prop_1) happen in the next 999 time units if (prop_2) in the next 500 time instants.\n\nLTL: (prop_1 imply ((prop_2 until [0, 300] prop_3) or ((prop_4 imply prop_2) and (prop_5 imply prop_6))))\nnatural language: If (prop_1) happens, then keep (prop_2) to be true until (prop_3) in the next 300 time units, otherwise, if (prop_4) then (prop_2) and if (prop_5) then (prop_6).\n\nLTL: (globally [0, 354] prop_1 and (prop_2 imply (globally [0, 521] (prop_3) and finally [521, 996] prop_4)))\nnatural language: Stay (prop_1) for 354 timesteps, and if (prop_2) happens, then first keep (prop_3) and then let (prop_4) happen at some point during 521 to 996 time steps.\n\nLTL: (finally [0, 1000] prop_1 and (prop_2 imply (prop_3 until [0, 500] prop_4)))\nnatural language: Manage to achieve (prop_1) in the next 1000 time steps, and if (prop_2) happens in this process, keep (prop_3) until (prop_4) for 500 time units.\n\nLTL: "
    
    prompt="Try to transform the signal temporal logic into natural languages, the operators in the signal temporal logic are: negation, imply, and, equal, until, globally, finally, or .\nThe examples are as following:\n\nLTL: (finally [317,767] (prop_3 equal prop_2) imply prop_1)\nnatural language: It is required that at a certain point within the next 317 to 767 time units the scenario in which ( prop_3 ) is equivalent to the scenario in which ( prop_2 ) happens , and only then ( prop_1 ) .\n\nLTL: (((prop_4 or prop_2) or prop_3) until prop_1)\nnatural language: In case that at some point ( prop_4 ) or ( prop_2 ) or ( prop_3 ) is detected and continued until then at some other point ( prop_1 ) should be detected as well .\n\nLTL: (((prop_2 until [417,741] prop_3) imply prop_1) or prop_4)\nnatural language: ( prop_2 ) should happen and hold until at a certain time point during the 417 to 741 time units the scenario that ( prop_3 ) should happen then ( prop_1 ) , or else ( prop_4 ) .\n\nLTL: ((prop_3 until [500,903] prop_1) and negation prop_2)\nnatural language:  ( prop_3 ) happens and continues until at some point during the 500 to 903 time units ( prop_1 ) happens , and in the same time ( prop_2 ) does not happen .\n\nLTL: (globally [107,513] prop_1 or (prop_3 and prop_2))\nnatural language: For each time instant in the next 107 to 513 time units ( prop_1 ) is true , or else ( prop_3 ) happens and ( prop_2 ) happens at the same time.\n\nLTL: ((prop_3 and prop_2) or (prop_4 until[469,961] prop_1))\nnatural language:  ( prop_3 ) and ( prop_2 ) should happen at the same time , or else ( prop_4 ) happens and continues until at some point during the 469 to 961 time units ( prop_1 ) happens .\n\nLTL: (negation prop_2 or (prop_1 until[286,348] prop_3))\nnatural language:  ( prop_2 ) should not happen , or else ( prop_1 ) happens and continues until at some point during the 286 to 348 time units ( prop_3 ) happens .\n\nLTL: ((prop_1 imply prop_2) and (prop_3 imply negation prop_2))\nnatural language: While (prop_1) , do (prop_2) , and when (prop_3) , stop (prop_2) .\n\nLTL: (prop_1 imply finally [300, infinite] prop_2)\nnatural language: If (prop_1) happens, then some time after the next 300 time steps (prop_2) should happen.\n\nLTL: (prop_1 imply (globally prop_2 and (prop_3 imply finally [0, 10] prop_4)))\nnatural language: If (prop_1) happens, then for all time afterward (prop_2) holds and if, in addition, if (prop_3) occurs, then (prop_4) eventually occurs in the next 10 time units.\n\nLTL: (prop_1 imply (negation prop_2 and (prop_3 until prop_4)))\nnatural language: If (prop_1), don't (prop_2), instead keep (prop_3) until (prop_4).\n\nLTL: (prop_4 imply ((prop_1 or prop_2) or prop_3))\nnatural language: If (prop_4), then make sure any of the following happens: (prop_1), (prop_2) or (prop_3).\n\nLTL: (finally [0, 500] (prop_2) imply finally [0, 999] prop_1)\nnatural language: Always make (prop_1) happen in the next 999 time units if (prop_2) in the next 500 time instants.\n\nLTL: (prop_1 imply ((prop_2 until [0, 300] prop_3) or ((prop_4 imply prop_2) and (prop_5 imply prop_6))))\nnatural language: If (prop_1) happens, then keep (prop_2) to be true until (prop_3) in the next 300 time units, otherwise, if (prop_4) then (prop_2) and if (prop_5) then (prop_6).\n\nLTL: (globally [0, 354] prop_1 and (prop_2 imply (globally [0, 521] (prop_3) and finally [521, 996] prop_4)))\nnatural language: Stay (prop_1) for 354 timesteps, and if (prop_2) happens, then first keep (prop_3) and then let (prop_4) happen at some point during 521 to 996 time steps.\n\nLTL: (finally [0, 1000] prop_1 and (prop_2 imply (prop_3 until [0, 500] prop_4)))\nnatural language: Manage to achieve (prop_1) in the next 1000 time steps, and if (prop_2) happens in this process, keep (prop_3) until (prop_4) for 500 time units.\n\nLTL: ((prop_2 and negation prop_3) until [50, 100] prop_1)\nnatural language: Until (prop_1) occurs within 50 to 100 timesteps, make sure that (prop_2) and not (prop_3) happen .\n\nLTL: (finally prop_2 and (negation prop_1 until prop_3))\nnatural language: Ensure that (prop_2) happens soon, but do not let (prop_1) occur until after (prop_3) .\n\nLTL: (prop_1 imply prop_2)\nnatural language: As long as (prop_1), make sure to maintain (prop_2) .\n\nLTL: ((prop_1 until prop_3) and (prop_4 imply prop_2))\nnatural language: Do (prop_1) until (prop_3), but once (prop_4) occurs then immediately (prop_2) .\n\nLTL:  (prop_2 imply finally [5, 12] (prop_1 or prop_3))\nnatural language: Once (prop_2) occurs, ensure that either (prop_1) or (prop_3) happen within 5 to 12 timesteps .\n\nLTL: ((prop_1 and prop_2) imply negation prop_3)\nnatural language: If you do (prop_1) and observe (prop_2), then you should not do (prop_3) .\n\nLTL: "
    + original_sent + '\nnatural language:',
    temperature=0.6,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  return response['choices'][0]['text'][1:]

# From generated LTL-1 to generate NL_1
dataset_nl_1 = [];
for i in range(len(list_ltl)):
  paraphrased_logic_sentence_raw = paraphrase_GPT3(pre_to_mid_exp(list_ltl[i])[0])
  for index in range(len(paraphrased_logic_sentence_raw )):
    if paraphrased_logic_sentence_raw[index] != '\n' and paraphrased_logic_sentence_raw[index] != ' ':
      break
  paraphrased_logic_sentence = paraphrased_logic_sentence_raw[index:]
  dataset_nl_1.append(paraphrased_logic_sentence)

# From LTL-2 to NL_2 and generate excel file
path = '../Raw_data'
f = open(path+'/test1.csv','w')
csv_writer = csv.writer(f)
csv_writer.writerow(['paraphrased_logic_sentence','logic_ltl_true_natural_order','original_logic_sentence', 'logic_ltl', 'Mark', 'Comments'])

for i in range(len(list_ltl)):
  csv_writer.writerow([dataset_nl_1[i],  pre_to_mid_exp(list_ltl[i])[0]
                       ,' ',list_ltl[i]
                       , str(1), ''])
f.close()

df = pd.read_csv(path+'/test1.csv')
df.to_excel(path+'/output_davinci_oneround_loop_'+str(filename)+'.xlsx', 'Sheet1')
