import transformers
import torch
from transformers import (AutoModelForSeq2SeqLM, 
                            AutoTokenizer)
import datasets
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
import csv
import streamlit as st
import datetime
import time
from PIL import Image
import mtl

output_dir = '../'
model_checkpoint = "t5-base"
prefix = "Transform the following sentence into Signal Temporal logic: "

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_input_length = 1024
max_target_length = 128

operation_list = ['negation', 'and', 'imply', 'equal', 'or', 'globally', 'finally', 'until', '(', ')']
two_leaves = ['and', 'imply', 'equal', 'or', 'until']
one_leaf = ['negation', 'globally', 'finally']
no_leaf = ['prop**']

dict_operation = {}
dict_operation['and'] = '&'; dict_operation['imply'] = '->'; dict_operation['equal'] = '<->'; dict_operation['or'] = '|'
dict_operation['globally'] = 'G'; dict_operation['finally'] = 'F'; dict_operation['until'] = 'U'; dict_operation['negation'] = '~'

def item_is_time(item_input):
  judge = False
  if item_input[0]=='[' and item_input[-1]==']':
    try:
      item_list = item_input.split(',')
    except:
      pass
    if len(item_list) == 2 and item_list[0][-1] != [' '] and item_list[1][0] != [' ']:
      judge = True
  return judge

def item_token_correct(my_list):
  error = ''
  item_count = 0
  for i, item in enumerate(my_list):
    if item == '':
      if i == 0:
        error = 'Spacing error, there are extra space at the beginning of TL'
      else:
        error = 'Spacing error, there are extra space after ' + my_list[i-1]
    elif item in operation_list or item_is_time(item) or item[0:5] == 'prop_':
      pass
    else:
      error = 'Syntex error in the token '+ item
  return error

def word2operation(input_list):
  post_list = []
  for i, item in enumerate(input_list):
    if item[0]=='[' and item[-1]==']':
      post_list.pop(-1)
      post_list.append(dict_operation[input_list[i-1]]+'[0,200]')
      #post_list.append(input_list[i-1]+input_list[i])
    elif item == '(' or item == ')' or item[0:5]=='prop_':
      post_list.append(item)
    else:
      post_list.append(dict_operation[item])
  return post_list

def prop_count(input_list):
  error = ''
  for i in range(1, 8):
    if input_list.count('prop_'+str(i)) > 1:
      error = 'prop_' + str(i)
  return error

def parenthe_matching_index(input_list):
  count_stack = 0
  for i, item in enumerate(input_list):
    if item == '(':
      count_stack += 1
    elif item ==')':
      count_stack -= 1
      if count_stack == 0:
        break
  return i

def outer_parentheses_check(input_list):
  output_list = input_list
  if input_list[0] != '(' or input_list[-1] != ')' or parenthe_matching_index(input_list) != len(input_list) -1:
    output_list = ['('] + input_list + [')']
  return output_list

def judge_synthetic(my_list):
  error = ''
  if my_list.count('(') != my_list.count(')'): # Check the parentheses matching
    error = 'Parentheses not matching!'
  elif item_token_correct(my_list) != '': # Check token correctness
    error = item_token_correct(my_list)
  elif prop_count(my_list) != '':
    error = 'Multiple ' + prop_count(my_list) + ' in the same TL!'
  else:
    my_list = outer_parentheses_check(my_list)
    try:
      mtl.parse(' '.join(word2operation(my_list))) # Check parsing correctness
    except:
      error = 'Parsing error. The TL structure is wrong!'
  return error

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["logic_sentence"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    model_inputs["logic_sentence"] = examples["logic_sentence"]
    model_inputs["id"] = examples["id"]
    return model_inputs

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained(output_dir+"t5-base-epoch20-infix-word-03-26-64220/checkpoint-72000").to(device)
home_path_output = '/home/ycchen/LTL/web'

def NL2TL(test_sentence):
    f = open(home_path_output+'/total_data.csv','w')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['id', 'logic_ltl', 'logic_sentence'])
    csv_writer.writerow([0, '', test_sentence])
    f.close()

    dataset = load_dataset('csv', data_files=home_path_output + '/total_data.csv')
    test_dataset= dataset['train']
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)


    for i in range(len(tokenized_test_dataset)):
        inputs = [prefix + tokenized_test_dataset[i]['logic_sentence']]

    inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt").to(device)
    output = model.generate(**inputs, num_beams=8, do_sample=True, max_length=max_target_length)
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return decoded_output

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():
    #st.write("Here goes your normal Streamlit app...")
   # st.button("Click me")

    image = Image.open('logos/1280px-MIT_logo.svg')
    st.image(image, width=120)

    st.title("NL2TL")
    st.markdown("<h1 style='color: red; font-size: 22px;'>Translating Natural Language (NL) into Temporal Logic (TL).</h1>", unsafe_allow_html=True)
    st.info('To enhance generalization, atomic propositions (AP) are hidden as (prop_i), and each (prop_i) appears only once in TL', icon="🔥")
    st.info('The users should first input NL and press Translate to see the predicted TL, if TL is not correct, please fill in the correct one and press Submit', icon="🔥")
    st.info(
        'For the submitted TL, please be careful that there should be space between each token',
        icon="🔥")
    st.info('The NL instructions can be like: \n'
            '1) If at some point ( prop_1 ) is equivalent to ( prop_4 ) and this scenario continues until at some other point ( prop_3 ) is detected , or else ( prop_2 ) .\n'
            '🤖 ( ( ( prop_1 equal prop_4 ) until prop_3 ) or prop_2 )\n'
            '2) ( prop_3 ) happens if and only if ( prop_4 ) is true.\n'
            '🤖 ( prop_3 equal prop_4 )\n'
            '3) Wait till (prop_1) is complete, then, (prop_2) for next 20 units and (prop_3) for next 40 units.\n'
            '🤖 ( finally prop_1 imply ( globally [0, 20] prop_2 and globally [0, 40] prop_3 ) )', icon="🔥")
    st.info('Codes, data, specific illustration ,and more examples at https://github.com/yongchao98/NL2TL', icon="🔥")
    #buff, col, buff2 = st.columns([1,3,1])
    #col.text_input('smaller text window:')

    user_name = st.text_input('Input your name if possible:', 'Anonymous')
    comments= st.text_input('Input possible constructive comments:', 'e.g., the model performs poorly when STLs are short')
    if st.button('Submit comments'):
        with open('comments.txt', 'a') as file:
            if comments != 'e.g., the model performs poorly when STLs are short':
                file.write(comments + '\n')
                st.success('Your comments have been recorded!', icon="✅")
    test_sentence = st.text_area('Input NL Instruction:', 'First, (prop_1) and then wait for 10 units and then (prop_2), and (prop_3) also if feasible.', height=120)
    correct_sentence = ''
    translation = NL2TL(test_sentence)

    if st.button('Translate'):
        st.write('Translated Signal TL:')
        st.write(translation)

    correct_sentence = st.text_input('Input corrected TL:')
    if st.button('Submit'):
        st.markdown(
            "<h1 style='color: blue; font-size: 18px;'>The input NL is: </h1>",
            unsafe_allow_html=True)
        st.write(test_sentence)
        st.markdown(
            "<h1 style='color: blue; font-size: 18px;'>The predicted TL is: </h1>",
            unsafe_allow_html=True)
        st.write(translation)
        st.markdown(
            "<h1 style='color: blue; font-size: 18px;'>The corrected TL is: </h1>",
            unsafe_allow_html=True)
        st.write(correct_sentence)
        with open('user_data1_03_06.txt', 'a') as file:
            file.seek(0, 2)  # move the file pointer to the end of the file
            if judge_synthetic(correct_sentence.split(' ')) == '':
                file.write('User name: ' + user_name + '\n')
                file.write(str(datetime.datetime.now()) + ' / ' + str(time.time()) + '\n')
                file.write('input NL: ' + test_sentence + '\n')
                file.write('predicted TL: ' + translation + '\n')
                file.write('corrected TL: ' + correct_sentence + '\n\n')
                st.success('Data contribution success!', icon="✅")
            else:
                e = RuntimeError(judge_synthetic(correct_sentence.split(' ')))
                st.exception(e)

