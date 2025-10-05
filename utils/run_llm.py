# CUDA_VISIBLE_DEVICES=0 python 05_run_llm_on_P-SET.py --attribute_output True
# CUDA_VISIBLE_DEVICES=1 python 05_run_llm_on_P-SET.py --attribute_output False
# python -i 05_run_llm_on_P-SET.py --attribute_output All
# nohup ollama serve > ollama.log 2>&1 &
#  srun --partition=gpu_h100 --gres=gpu:1 --cpus-per-task=16 --mem=100G --time=8:00:00 --pty bash -i
# sudo kill -9 $(nvidia-smi | awk 'NR>8 {print $5}' | grep -E '^[0-9]+$')
import pandas as pd
import os, sys
from datetime import datetime
from itertools import product
from glob import glob

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from utils.event_extractor import EventExtractor
from utils.evaluation_samples import focus_ids as evaluation_focus_ids
from config import event_types, event_descriptions

import streamlit as st

analysis_types = ['Sent','Doc']
kw_input_types = [False]
ex_input_types = [True]
event_types_local = ['Sleep','Excretion','Eating','Family','Pain'][:1]
dataset = 'P-SET'
prompt_version = 4
print(f"Prompt Version {prompt_version}")

attribute_output = True

input_to_analyse = [st.text_input("Enter Text Fragment","")]
llm_type=st.text_input("LLM_Type","llama3.1:70b")
evidence={'keywords':[], 'event_names':[], 'dct':[('2167-05-18 05:26:00','2167-05-18 05:33:00')]}
prompt_version = st.selectbox("Prompt version",options=[2,3,4],index=2)
keyword_input = st.radio("Keyword_input",options=[True, False], index=1)
example_input = st.radio("Example_input",options=[True, False], index=1)
event_extractor_object = EventExtractor(event_name_model_type="llm",attribute_model_type="None",llm_type=llm_type)
if st.button("Run LLM"):
    res  = event_extractor_object.extract_events(texts=input_to_analyse,                                                                                                               
                                                event_names=event_types, 
                                                event_descriptions=event_descriptions, 
                                                prompt_version=prompt_version,
                                                prompt_evidence=evidence, 
                                                attribute_output=attribute_output,
                                                keyword_input=keyword_input, 
                                                example_input=example_input,
                                                )
    st.write(res.event_list[0].raw_output)