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

#init_num = 10
#filename = 0

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

def generate_ltl():
  two_leaves = ['&', '->', '<->', '|', 'U', 'U[*,**']
  one_leaf = ['negation', 'G', 'F', 'G[*,**', 'F[*,**']
  random_num = random.randint(2, 6)
  num_prop = [i for i in range(1,random_num)]
  random.shuffle(num_prop)
  base_list = ['prop_'+str(item) for item in num_prop]
  if len(base_list)==1:
    random_num_2 = random.randint(0, len(one_leaf)-1)
    if one_leaf[random_num_2] in prop_with_time:
      time1 = random.randint(1, 500); time2 = random.randint(1, 500) + time1
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
  ltl_can = generate_ltl_v2(6)
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
      print(ltl_can)
      word_express, operation_express = pre_to_mid_exp(ltl_can)
      print(word_express)
      print(operation_express)
      print('\n')

# Generate NL-1 from STL-1
def paraphrase_GPT3(original_sent):
  openai.api_key = ''  # input your openai key
  response = openai.Completion.create(
    model="text-davinci-002",

    # Prompt for predicting NL from LTL
    #prompt="Try to transform the signal temporal logic into natural languages, the operators in the signal temporal logic are: negation, imply, and, equal, until, globally, finally, or .\nThe examples are as following:\nLTL: (((prop_2 equal prop_3) and prop_4) equal prop_1)\nnatural language: If ( prop_2 ) is equivalent to ( prop_3 ) and also ( prop_4 ) , then the above scenario is equivalent to ( prop_1 ) .\n\nLTL: (globally [145,584] (prop_1 or prop_2) or prop_3)\nnatural language: For each time instant in the coming 145 to 584 time units either ( prop_1 ) or ( prop_2 ) should be detected , or else ( prop_3 ) .\n\nLTL: (finally [317,767] (prop_3 equal prop_2) imply prop_1)\nnatural language: It is required that at a certain point within the next 317 to 767 time units the scenario in which ( prop_3 ) is equivalent to the scenario in which ( prop_2 ) happens , and only then ( prop_1 ) .\n\nLTL: (((prop_4 or prop_2) or prop_3) until prop_1)\nnatural language: In case that at some point ( prop_4 ) or ( prop_2 ) or ( prop_3 ) is detected and continued until then at some other point ( prop_1 ) should be detected as well .\n\nLTL: (((prop_2 until [417,741] prop_3) imply prop_1) or prop_4)\nnatural language: ( prop_2 ) should happen and hold until at a certain time point during the 417 to 741 time units the scenario that ( prop_3 ) should happen then ( prop_1 ) , or else ( prop_4 ) .\n\nLTL: (globally [184,440] (negation (prop_1 or prop_3)) imply prop_2)\nnatural language: For each time instant in the next 184 to 440 time units if it is not the case that ( prop_1 ) or ( prop_3 ) then ( prop_2 ) .\n\nLTL: (((prop_3 until [391,525] prop_2) and prop_1) and prop_4)\nnatural language: In case that ( prop_3 ) continues to happen until at some point during the first 391 to 525 time units that ( prop_2 ) happens , as well as ( prop_1 ) , and ( prop_4 ) then .\n\nLTL: ((finally (negation prop_1) imply prop_2) imply prop_3)\nnatural language: If finally that ( prop_1 ) is not detected then ( prop_2 ) , then ( prop_3 ) .\n\nLTL: (negation (prop_1 equal prop_2) until [394,530] prop_3)\nnatural language: It is not the case that ( prop_1 ) if and only if ( prop_2 ) is true , the above scenario will hold until ( prop_3 ) will be detected at some time point during the next 394 to 530 time units .\n\nLTL: (((prop_1 or prop_3) imply prop_4) until [193,266] prop_2)\nnatural language:  If at some point ( prop_1 ) or ( prop_3 ) then ( prop_4 ) happens and this scenario will hold until at some other point during the 193 to 266 time units ( prop_2 ) is detected .\n\nLTL: (negation (prop_1 until [77,432] prop_2) and prop_3)\nnatural language:  It is not the case that ( prop_1 ) happens and continues to happen until at some point during the 77 to 432 time units ( prop_2 ) is detected , and ( prop_3 ) .\n\nLTL: "
    prompt="Try to transform the signal temporal logic into natural languages, the operators in the signal temporal logic are: negation, imply, and, equal, until, globally, finally, or .\nThe examples are as following:\nLTL: (((prop_2 equal prop_3) and prop_4) equal prop_1)\nnatural language: If ( prop_2 ) is equivalent to ( prop_3 ) and also ( prop_4 ) , then the above scenario is equivalent to ( prop_1 ) .\n\nLTL: (globally [145,584] (prop_1 or prop_2) or prop_3)\nnatural language: For each time instant in the coming 145 to 584 time units either ( prop_1 ) or ( prop_2 ) should be detected , or else ( prop_3 ) .\n\nLTL: (finally [317,767] (prop_3 equal prop_2) imply prop_1)\nnatural language: It is required that at a certain point within the next 317 to 767 time units the scenario in which ( prop_3 ) is equivalent to the scenario in which ( prop_2 ) happens , and only then ( prop_1 ) .\n\nLTL: (((prop_4 or prop_2) or prop_3) until prop_1)\nnatural language: In case that at some point ( prop_4 ) or ( prop_2 ) or ( prop_3 ) is detected and continued until then at some other point ( prop_1 ) should be detected as well .\n\nLTL: (((prop_2 until [417,741] prop_3) imply prop_1) or prop_4)\nnatural language: ( prop_2 ) should happen and hold until at a certain time point during the 417 to 741 time units the scenario that ( prop_3 ) should happen then ( prop_1 ) , or else ( prop_4 ) .\n\nLTL: (globally [184,440] (negation (prop_1 or prop_3)) imply prop_2)\nnatural language: For each time instant in the next 184 to 440 time units if it is not the case that ( prop_1 ) or ( prop_3 ) then ( prop_2 ) .\n\nLTL: (((prop_3 until [391,525] prop_2) and prop_1) and prop_4)\nnatural language: In case that ( prop_3 ) continues to happen until at some point during the first 391 to 525 time units that ( prop_2 ) happens , as well as ( prop_1 ) , and ( prop_4 ) then .\n\nLTL: ((finally (negation prop_1) imply prop_2) imply prop_3)\nnatural language: If finally that ( prop_1 ) is not detected then ( prop_2 ) , then ( prop_3 ) .\n\nLTL: (negation (prop_1 equal prop_2) until [394,530] prop_3)\nnatural language: It is not the case that ( prop_1 ) if and only if ( prop_2 ) is true , the above scenario will hold until ( prop_3 ) will be detected at some time point during the next 394 to 530 time units .\n\nLTL: (((prop_1 or prop_3) imply prop_4) until [193,266] prop_2)\nnatural language:  If at some point ( prop_1 ) or ( prop_3 ) then ( prop_4 ) happens and this scenario will hold until at some other point during the 193 to 266 time units ( prop_2 ) is detected .\n\nLTL: (negation (prop_1 until [77,432] prop_2) and prop_3)\nnatural language:  It is not the case that ( prop_1 ) happens and continues to happen until at some point during the 77 to 432 time units ( prop_2 ) is detected , and ( prop_3 ) .\n\nLTL: ((prop_2 and prop_4) or (prop_3 until prop_1))\nnatural language: It is required that both ( prop_2 ) and ( prop_4 ) happen at the same time, or else ( prop_3 ) happens and continues until ( prop_1 ) happens.\n\nLTL: ((prop_3 until [500,903] prop_1) and negation prop_2)\nnatural language:  ( prop_3 ) happens and continues until at some point during the 500 to 903 time units ( prop_1 ) happens , and in the same time ( prop_2 ) does not happen .\n\nLTL: (globally [107,513] prop_1 or (prop_3 and prop_2))\nnatural language: For each time instant in the next 107 to 513 time units ( prop_1 ) is true , or else ( prop_3 ) happens and ( prop_2 ) happens at the same time.\n\nLTL: ((prop_1 or prop_2) until [142,365] (prop_4 and prop_3))\nnatural language:  ( prop_1 ) or ( prop_2 ) happens and continues until at some point during the 142 to 365 time units ( prop_4 ) happens and ( prop_3 ) happens at the same time .\n\nLTL: (globally [91,471] prop_2 and (prop_1 or prop_3))\nnatural language:  For each time instant in the next 91 to 471 time units ( prop_2 ) happens , and ( prop_1 ) or ( prop_3 ) also happens .\n\nLTL: ((negation (prop_1) equal prop_2) imply globally [483,715] prop_3)\nnatural language:  If the case ( prop_1 ) does not happen is equivalent to the case ( prop_2 ) happens , then for each time instant in the next 483 to 715 time units ( prop_3 ) is true .\n\nLTL: ((prop_1 or prop_3) and negation prop_2)\nnatural language: It is required that either ( prop_1 ) or ( prop_3 ) happens , and in the same time ( prop_2 ) does not happen .\n\nLTL: (globally [320,493] prop_2 equal (prop_3 imply prop_1))\nnatural language:  For each time instant in the next 320 to 493 time units ( prop_2 ) happens , is equivalent to the case that if ( prop_3 ) then ( prop_1 ) .\n\nLTL: ((prop_1 or prop_2) until [152,154] negation prop_3)\nnatural language: ( prop_1 ) or ( prop_2 ) happens and continues until at some point during the 152 to 154 time units that ( prop_3 ) does not happen .\n\nLTL: ((negation (prop_1) and prop_2) equal finally [230,280] prop_3)\nnatural language: ( prop_1 ) should not happen and ( prop_2 ) should happen at the same time , and the above scenario is equivalent to the case that at some point during the 230 to 280 time units ( prop_3 ) happens .\n\nLTL: ((prop_2 imply prop_3) and finally [7,283] prop_1)\nnatural language:  If ( prop_2 ) then ( prop_3 ) happens , and at some point during the 7 to 283 time units ( prop_1 ) happens .\n\nLTL: ((prop_3 and prop_2) or (prop_4 until[469,961] prop_1))\nnatural language:  ( prop_3 ) and ( prop_2 ) should happen at the same time , or else ( prop_4 ) happens and continues until at some point during the 469 to 961 time units ( prop_1 ) happens .\n\nLTL: (negation prop_2 or (prop_1 until[286,348] prop_3))\nnatural language:  ( prop_2 ) should not happen , or else ( prop_1 ) happens and continues until at some point during the 286 to 348 time units ( prop_3 ) happens .\n\nLTL: "
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

# From generated NL-1 to LTL-2
def word2operation(pre_order_list):
  dict_operation = {}
  dict_operation['and'] = '&'; dict_operation['imply'] = '->'; dict_operation['equal'] = '<->'; dict_operation['or'] = '|'
  dict_operation['globally'] = 'G'; dict_operation['finally'] = 'F'; dict_operation['until'] = 'U'; dict_operation['negation'] = 'negation'
  post_list = []
  for item in pre_order_list:
    if item[0:9] == 'globally ':
      post_list.append('G'+item[9:])
    elif item[0:8] == 'finally ':
      post_list.append('F'+item[8:])
    elif item[0:6] == 'until ':
      post_list.append('U'+item[6:])
    elif match(item, 'prop**'):
      post_list.append(item)
    else:
      post_list.append(dict_operation[item])
  return post_list

def judge_leaf_num_2(item):
  two_leaves = ['and', 'imply', 'equal', 'or', 'until']
  one_leaf = ['globally', 'finally', 'negation']
  if item in two_leaves or item[0:5]=='until':
    return 2
  elif match(item, 'prop**'):
    return 0
  else:
    return 1

def pre_to_mid_exp_2(item_list_original):
  item_list = copy.deepcopy(item_list_original)
  item_list.reverse()
  word_list = word2operation(item_list)
  stack_list = []; stack_list_2 = []
  mark = 0
  for i,item in enumerate(item_list):
    if judge_leaf_num_2(item) == 0:
      stack_list.append(word_list[i])
      stack_list_2.append(item_list[i])
    elif judge_leaf_num_2(item) == 1:
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
    elif judge_leaf_num_2(item) == 2:
      candidate = '('+ stack_list[-1] + ' ' + word_list[i] + ' ' + stack_list[-2] + ')'
      candidate_2 = '('+ stack_list_2[-1] + ' ' + item_list[i] + ' ' + stack_list_2[-2] + ')'
      stack_list.pop(-1)
      stack_list.pop(-1)
      stack_list.append(candidate)
      stack_list_2.pop(-1)
      stack_list_2.pop(-1)
      stack_list_2.append(candidate_2)
      mark=0
  return candidate_2, candidate

def paraphrase_GPT3_to_ltl(original_sent):
  openai.api_key = ''  # input your openai key

  response = openai.Completion.create(
    model="text-davinci-002",

    # Prompt for transforming natural language into LTL with prefix order like ['U', '&', '<->', 'prop_1', 'prop_4', 'prop_2', 'prop_3']
    #prompt="Try to transform the following natural languages into signal temporal logics, the operators in the signal temporal logic are: negation, imply, and, equal, until, globally, finally, or .\nThe signal temporal logics are prefix expressions. The examples are as following:\nnatural language: It is required that for every moment during the interval 489 to 663 either the event that ( prop_1 ) is detected and in response ( prop_3 ) should happen , or ( prop_2 ) should be true .\nSTL: ['or', 'globally [489,663]', 'imply', 'prop_1', 'prop_3', 'prop_2']\n\nnatural language: It should be the case that if ( prop_4 ) or ( prop_2 ) then ( prop_3 ), and ( prop_1 ) .\nSTL: ['and', 'imply', 'or', 'prop_4', 'prop_2', 'prop_3', 'prop_1']\n\nnatural language: It is always the case that if it is not the case that ( prop_2 ) then ( prop_3 ), and ( prop_1 ) .\nSTL: ['and', 'globally', 'imply', 'negation', 'prop_2', 'prop_3', 'prop_1']\n\nnatural language: ( prop_3 ) should happen until at some point during the 483 to 907 time units , then ( prop_1 ) should happen, or else ( prop_2 ) , or else ( prop_4 ) .\nSTL: ['or', 'or', 'until [483,907]', 'prop_3', 'prop_1', 'prop_2', 'prop_4']\n\nnatural language: It is true that if the scenario in which ( prop_4 ) leads to ( prop_3 ) happens and continues until ( prop_1 ) happens , then ( prop_2 ) should be observed . And it is also true that if ( prop_2 ) is observed , then ( prop_4 ) should have led to ( prop_3 ) and this condition continues until ( prop_1 ) happens .\nSTL: ['equal', 'until', 'imply', 'prop_4', 'prop_3', 'prop_1', 'prop_2']\n\nnatural language: Before a certain time point within the next 15 to 196 time units ( prop_2 ) leads to ( prop_4 ) and ( prop_3 ) is true , then starting from this time point ( prop_1 ) .\nSTL: ['until [15,196]', 'and', 'imply', 'prop_2', 'prop_4', 'prop_3', 'prop_1']\n\nnatural language: If ( prop_4 ) then implies ( prop_2 ), and in the same time ( prop_1 ) , or else ( prop_3 ) .\nSTL:  ['or', 'and', 'imply', 'prop_4', 'prop_2', 'prop_1', 'prop_3']\n\nnatural language: It is always the case that if within the next 139 to 563 time units , the scenario that ( prop_2 ) is detected then as a response ( prop_1 ) , and ( prop_3 ) .\nSTL:  ['and', 'globally [139,563]', 'imply', 'prop_2', 'prop_1', 'prop_3']\n\nnatural language: If it is the case that ( prop_2 ) and ( prop_4 ) are equivalent and continue to happen until the scenario that ( prop_1 ) is detected then in response ( prop_3 ) should happen .\nSTL:  ['imply', 'until', 'equal', 'prop_2', 'prop_4', 'prop_1', 'prop_3']\n\nnatural language: If ( prop_3 ) then implies ( prop_4 ), this condition should continue to happen until at some point within the next 450 to 942 time units , after that ( prop_2 ) , or ( prop_1 ) .\nSTL:  ['or', 'until [450,942]', 'imply', 'prop_3', 'prop_4', 'prop_2', 'prop_1']\n\nnatural language: "
    prompt="Try to transform the following natural languages into signal temporal logics, the operators in the signal temporal logic are: negation, imply, and, equal, until, globally, finally, or .\nThe signal temporal logics are prefix expressions. The examples are as following:\nnatural language: It is required that for every moment during the interval 489 to 663 either the event that ( prop_1 ) is detected and in response ( prop_3 ) should happen , or ( prop_2 ) should be true .\nSTL: ['or', 'globally [489,663]', 'imply', 'prop_1', 'prop_3', 'prop_2']\n\nnatural language: It should be the case that if ( prop_4 ) or ( prop_2 ) then ( prop_3 ), and ( prop_1 ) .\nSTL: ['and', 'imply', 'or', 'prop_4', 'prop_2', 'prop_3', 'prop_1']\n\nnatural language: It is always the case that if it is not the case that ( prop_2 ) then ( prop_3 ), and ( prop_1 ) .\nSTL: ['and', 'globally', 'imply', 'negation', 'prop_2', 'prop_3', 'prop_1']\n\nnatural language: ( prop_3 ) should happen until at some point during the 483 to 907 time units , then ( prop_1 ) should happen, or else ( prop_2 ) , or else ( prop_4 ) .\nSTL: ['or', 'or', 'until [483,907]', 'prop_3', 'prop_1', 'prop_2', 'prop_4']\n\nnatural language: It is true that if the scenario in which ( prop_4 ) leads to ( prop_3 ) happens and continues until ( prop_1 ) happens , then ( prop_2 ) should be observed . And it is also true that if ( prop_2 ) is observed , then ( prop_4 ) should have led to ( prop_3 ) and this condition continues until ( prop_1 ) happens .\nSTL: ['equal', 'until', 'imply', 'prop_4', 'prop_3', 'prop_1', 'prop_2']\n\nnatural language: Before a certain time point within the next 15 to 196 time units ( prop_2 ) leads to ( prop_4 ) and ( prop_3 ) is true , then starting from this time point ( prop_1 ) .\nSTL: ['until [15,196]', 'and', 'imply', 'prop_2', 'prop_4', 'prop_3', 'prop_1']\n\nnatural language: If ( prop_4 ) then implies ( prop_2 ), and in the same time ( prop_1 ) , or else ( prop_3 ) .\nSTL:  ['or', 'and', 'imply', 'prop_4', 'prop_2', 'prop_1', 'prop_3']\n\nnatural language: It is always the case that if within the next 139 to 563 time units , the scenario that ( prop_2 ) is detected then as a response ( prop_1 ) , and ( prop_3 ) .\nSTL:  ['and', 'globally [139,563]', 'imply', 'prop_2', 'prop_1', 'prop_3']\n\nnatural language: If it is the case that ( prop_2 ) and ( prop_4 ) are equivalent and continue to happen until the scenario that ( prop_1 ) is detected then in response ( prop_3 ) should happen .\nSTL:  ['imply', 'until', 'equal', 'prop_2', 'prop_4', 'prop_1', 'prop_3']\n\nnatural language: If ( prop_3 ) then implies ( prop_4 ), this condition should continue to happen until at some point within the next 450 to 942 time units , after that ( prop_2 ) , or ( prop_1 ) .\nSTL:  ['or', 'until [450,942]', 'imply', 'prop_3', 'prop_4', 'prop_2', 'prop_1']\n\nnatural language: ( prop_3 ) happens until a time in the next 5 to 12 units that ( prop_4 ) does not happen .\nSTL:  ['until [5,12]', 'prop_3', 'negation', 'prop_4']\n\nnatural language: The time that ( prop_3 ) happens is when ( prop_1 ) happens , and vice versa .\nSTL:  ['equal', 'prop_3', 'prop_1']\n\nnatural language: It is required that both ( prop_2 ) and ( prop_4 ) happen at the same time, or else ( prop_3 ) happens and continues until ( prop_1 ) does not happen.\nSTL:  ['or', 'and', 'prop_2', 'prop_4', 'until', 'prop_3', 'negation', 'prop_1']\n\nnatural language: ( prop_3 ) happens and continues until at some point during the 500 to 903 time units ( prop_1 ) happens , and in the same time ( prop_2 ) does not happen .\nSTL:  ['and', 'until [500,903]', 'prop_3', 'prop_1', 'negation', 'prop_2']\n\nnatural language: For each time instant in the next 107 to 513 time units ( prop_1 ) is true , or else ( prop_3 ) happens and ( prop_2 ) happens at the same time.\nSTL:  ['or', 'globally [107,513]', 'prop_1', 'and', 'prop_3', 'prop_2']\n(globally [107,513] prop_1 or (prop_3 and prop_2))\n\nnatural language: ( prop_1 ) or ( prop_2 ) happens and continues until at some point during the 142 to 365 time units ( prop_4 ) happens and ( prop_3 ) happens at the same time .\nSTL:  ['until [142,365]', 'or', 'prop_1', 'prop_2', 'and', 'prop_4', 'prop_3']\n\nnatural language:  For each time instant in the next 91 to 471 time units ( prop_2 ) happens , and ( prop_1 ) or ( prop_3 ) also happens .\nSTL:  ['and', 'globally [91,471]', 'prop_2', 'or', 'prop_1', 'prop_3']\n\nnatural language: If the case ( prop_1 ) does not happen is equivalent to the case ( prop_2 ) happens , then for each time instant in the next 483 to 715 time units ( prop_3 ) is true .\nSTL:  ['imply', 'equal', 'negation', 'prop_1', 'prop_2', 'globally [483,715]', 'prop_3']\n\nnatural language: It is required that either ( prop_1 ) or ( prop_3 ) happens , and in the same time ( prop_2 ) does not happen .\nSTL:  ['and', 'or', 'prop_1', 'prop_3', 'negation', 'prop_2']\n\nnatural language:  For each time instant in the next 320 to 493 time units ( prop_2 ) happens , is equivalent to the case that if ( prop_3 ) then ( prop_1 ) .\nSTL:  ['equal', 'globally [320,493]', 'prop_2', 'imply', 'prop_3', 'prop_1']\n\nnatural language: ( prop_1 ) or ( prop_2 ) happens and continues until at some point during the 152 to 154 time units that ( prop_3 ) does not happen .\nSTL:  ['until [152,154]', 'or', 'prop_1', 'prop_2', 'negation', 'prop_3']\n\nnatural language: ( prop_1 ) should not happen and ( prop_2 ) should happen at the same time , and the above scenario is equivalent to the case that at some point during the 230 to 280 time units ( prop_3 ) happens .\nSTL:  ['equal', 'and', 'negation', 'prop_1', 'prop_2', 'finally [230,280]', 'prop_3']\n\nnatural language:  If ( prop_2 ) then ( prop_3 ) happens , and at some point during the 7 to 283 time units ( prop_1 ) happens .\nSTL:  ['and', 'imply', 'prop_2', 'prop_3', 'finally [7,283]', 'prop_1']\n\nnatural language:  ( prop_3 ) and ( prop_2 ) should happen at the same time , or else ( prop_4 ) happens and continues until at some point during the 469 to 961 time units ( prop_1 ) happens .\nSTL:  ['or', 'and', 'prop_3', 'prop_2', 'until [469,961]', 'prop_4', 'prop_1']\n\nnatural language: ( prop_1 ) implies ( prop_3 ) , and ( prop_4 ) happens if and only if ( prop_2 ) .\nSTL:  ['and', 'equal', 'imply', 'prop_1', 'prop_3', 'prop_4', 'prop_2']\n\nnatural language: In the following 10 time steps , the ( prop_1 ) should always happen , and in the meantime , ( prop_2 ) should happen at least once .\nSTL:  ['and', 'globally [0,10]', 'prop_1', 'finally', 'prop_2']\n\nnatural language: ( prop_1 ) should not happen if ( prop_2 ) does not happen , and ( prop_3 ) should also be true all the time .\nSTL:  ['and', 'imply', 'negation', 'prop_2', 'negation', 'prop_1', 'globally', 'prop_3']\n\nnatural language: If ( prop_1 ) and ( prop_2 ), then ( prop_3 ) until ( prop_4 ) does not happen , and ( prop_5 ) until ( prop_6 ) does not happen .\nSTL:  ['and', 'imply', 'and', 'prop_1', 'prop_2', 'until', 'prop_3', 'negation', 'prop_4', 'until', 'prop_5', 'negation', 'prop_6']\n\nnatural language: For each time instant in the next 0 to 120 units, do ( prop_1 ) if ( prop_2 ) , and if possible, ( prop_4 ) .\nSTL:  ['and', 'globally [0,120]', 'imply', 'prop_2', 'prop_1', 'prop_4']\n\nnatural language: In the next 0 to 5 time units , do the ( prop_1 ) , but in the next 3 to 4 time units , ( prop_2 ) should not happen .\nSTL:  ['and', 'globally [0,5]', 'prop_1', 'globally [3,4]', 'negation', 'prop_2']\n\nnatural language: "
    + original_sent + '\nLTL:',

    temperature=0.6,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  return response['choices'][0]['text'][1:]

def verify_correct(test_list):
  two_leaves = ['and', 'imply', 'equal', 'or', 'until']
  one_leaf = ['globally', 'finally', 'negation']
  count = 1; mark =1; time_first_appear_index = -1
  for i, item in enumerate(test_list):
    if item in two_leaves or item[0:5] == 'until':
      count += 1
    elif item in one_leaf or item[0:8] == 'globally' or item[0:7] == 'finally':
      count += 0
    elif match(item, 'prop**'):
      count -= 1
      if time_first_appear_index == -1:
        time_first_appear_index = i
    else: mark =0
  if count !=0:
    mark = 0
  for i in range(time_first_appear_index, len(test_list)):
    if not match(test_list[i], 'prop**'):
      mark = 0
  return mark

def str2list_ltl_2(original_ltl):
  original_ltl = original_ltl[1:len(original_ltl)-1].split(', ')
  logic_ltl = []
  for item in original_ltl:
    if item[-1] == ',':
      logic_ltl.append(item[1:len(item)-2])
    else:
      logic_ltl.append(item[1:len(item)-1])
  return logic_ltl

print('-'*40)
print('From NL-1 to LTL-2')
LTL_2 = []
for i in range(len(dataset_nl_1)):
  mark = 0; total_try = 0
  while mark ==0:
    paraphrased_logic_ltl_raw = paraphrase_GPT3_to_ltl(dataset_nl_1[i])
    for index in range(len(paraphrased_logic_ltl_raw )):
      if paraphrased_logic_ltl_raw[index] != '\n' and paraphrased_logic_ltl_raw[index] != ' ':
        break
    paraphrased_logic_ltl = paraphrased_logic_ltl_raw[index:]
    mark = verify_correct(str2list_ltl_2(paraphrased_logic_ltl))
    total_try += 1
    if total_try >1:
      break
  if mark:
    print(str2list_ltl_2(paraphrased_logic_ltl))
    word_express, operation_express = pre_to_mid_exp_2(str2list_ltl_2(paraphrased_logic_ltl))
    print(word_express)
    print('\n')
    LTL_2.append(str2list_ltl_2(paraphrased_logic_ltl))

# From LTL-2 to NL_2 and generate excel file
path = '../Raw_data'
f = open(path+'/test1.csv','w')
csv_writer = csv.writer(f)
csv_writer.writerow(['paraphrased_logic_sentence','logic_ltl_true_natural_order','original_logic_sentence', 'logic_ltl', 'Mark', 'Comments'])

for i in range(len(LTL_2)):
  paraphrased_logic_sentence_raw = paraphrase_GPT3(pre_to_mid_exp_2(LTL_2[i])[0])
  for index in range(len(paraphrased_logic_sentence_raw )):
    if paraphrased_logic_sentence_raw[index] != '\n' and paraphrased_logic_sentence_raw[index] != ' ':
      break
  paraphrased_logic_sentence = paraphrased_logic_sentence_raw[index:]
  csv_writer.writerow([paraphrased_logic_sentence,  pre_to_mid_exp_2(LTL_2[i])[0]
                       ,' ',LTL_2[i]
                       , str(1), ''])
f.close()

df = pd.read_csv(path+'/test1.csv')
df.to_excel(path+'/output_davinci_loop_'+str(filename)+'.xlsx', 'Sheet1')
