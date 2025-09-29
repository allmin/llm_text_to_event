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
from utils.evaluation_samples import focus_ids
from config import event_types, event_descriptions
import argparse



def get_col_suffix(keyword_input, example_input):
    col_suffix = "no"
    if keyword_input and example_input:
        col_suffix = "all"
    elif keyword_input and not example_input:
        col_suffix = "keyword"
    elif not keyword_input and example_input:
        col_suffix = "example"
    return col_suffix

def combine_lists(x):
    combined = []
    for item in x:
        if item:
            if isinstance(item, (list, tuple)):
                combined.extend(item)
            else:
                combined.append(item)
    combined = [i for i in combined if i]
    res = set(combined)
    return list(res)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--attribute_output',
    type=str,
    choices=['True', 'False', 'All'],
    default='True',
    help="Choose between 'True', 'False', or 'All'"
)

args = parser.parse_args()

mapping = {"True": True, "False": False, "All": "All"}
value = mapping[args.attribute_output]
print(value)

if value in [True, False]:
    attribute_output_raw = [args.attribute_output]
elif value == 'All':
    attribute_output_raw = [True, False]
print(f'attribute_output:{attribute_output_raw}, type:{type(attribute_output_raw)}')
dataset = 'P-SET'
prompt_version = 3
llm_type="llama3.1:70b"
for ET in ['Sleep','Excretion','Eating','Family','Pain'][:1]:    
    output_folder = f"../exports/05b_llm_{llm_type}_{dataset}_v{prompt_version}/{ET}"
    for attribute_output in attribute_output_raw:
        os.makedirs(f"{output_folder}", exist_ok=True)
        for analysis_type in ['Sent', 'Doc']:
            if analysis_type == 'Sent':
                id_type = 'UID'
            elif analysis_type == 'Doc':
                id_type = 'ROW_ID'
            try:
                file = glob(f"../exports/04b_groundtruth/{dataset}/Generated/{ET}*{analysis_type}*.pkl")[0]
            except:
                print(f"No file found for {ET}")
                continue
            df_date=pd.read_pickle("../exports/04_dictionary_features.pkl")
            df_date[id_type] = df_date[id_type].astype(str)
            df_date['CHARTTIME'] = pd.to_datetime(df_date['CHARTTIME'])
            df_date['STORETIME'] = pd.to_datetime(df_date['STORETIME'])
            id2charttime = df_date.groupby(id_type)['CHARTTIME'].min().to_dict()
            id2storetime = df_date.groupby(id_type)['STORETIME'].max().to_dict()          
            file_name = os.path.basename(file).strip(".pkl")
            df = pd.read_pickle(file)
            df[id_type] = df[id_type].astype(str)
            df['Event_Name'] = [tuple(i) for i in df['Event_Name']]
            df['Keyword'] = [tuple(i) for i in df['Keyword']]
            if "CHARTTIME" not in df.columns:
                df['CHARTTIME'] = df[id_type].map(id2charttime)
                df['STORETIME'] = df[id_type].map(id2storetime)
            else:
                df['CHARTTIME'] = pd.to_datetime(df['CHARTTIME'])
                df['STORETIME'] = pd.to_datetime(df['STORETIME'])
            df['DCT'] = [(r['CHARTTIME'], r['STORETIME']) for _,r in df.iterrows()]
            print(f"------------{df.STORETIME-df.CHARTTIME}")
            print(analysis_type, len(df))
            # df = df[df[id_type].isin(focus_ids[analysis_type])]
            print(analysis_type, len(df))
            if analysis_type == 'Sent':
                df_temp = df.copy()
                focus_type = "Sentences"
                gt_column = f"Sent_gt_{ET}"
                input_to_analyse = df_temp.Sentence.tolist()
                
                
            elif analysis_type == 'Doc':
                df_temp = df.copy()
                focus_type = "Documents"
                gt_column = f"Doc_gt_{ET}"
                input_to_analyse = df_temp.Document.tolist()
            
            print(f"Event Type: {ET} | Rows: {len(df_temp)} | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | File: {file_name}")
            print(f'attribute_output:{attribute_output}, Time Start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            evidence={'keywords':df_temp.Keyword.tolist(), 'event_names':df_temp.Event_Name.tolist(), 'dct':df_temp.DCT.tolist()}
            for keyword_input, example_input in [i for i in product([False],[True])]:
                print(f"keyword_input:{keyword_input}, example_input:{example_input}, attribute_output:{attribute_output}, analysis_type:{analysis_type}")
                col_suffix = get_col_suffix(keyword_input, example_input)
                event_extractor_object = EventExtractor(event_name_model_type="llm",attribute_model_type="None",llm_type=llm_type)
                df_temp.loc[:,f"LLM_Events_{col_suffix}_evidence_{analysis_type}"] = event_extractor_object.extract_events(texts=input_to_analyse, 
                                                                                                                    event_names=event_types, 
                                                                                                                    event_descriptions=event_descriptions, 
                                                                                                                    prompt_version=prompt_version,
                                                                                                                    prompt_evidence=evidence, 
                                                                                                                    attribute_output=attribute_output,
                                                                                                                    keyword_input=keyword_input, 
                                                                                                                    example_input=example_input,
                                                                                                                    )
                df_temp.loc[:,f"Event_Name_LLM_Events_{col_suffix}_evidence_{analysis_type}"] = df_temp[f"LLM_Events_{col_suffix}_evidence_{analysis_type}"].apply(lambda x: x['event'])
                df_temp.loc[:,f"Attribute_LLM_Events_{col_suffix}_evidence_{analysis_type}"] = df_temp[f"LLM_Events_{col_suffix}_evidence_{analysis_type}"].apply(lambda x: x['attributes'])
                df_temp.loc[:,f"Text_Quote_LLM_Events_{col_suffix}_evidence_{analysis_type}"] = df_temp[f"LLM_Events_{col_suffix}_evidence_{analysis_type}"].apply(lambda x: x['text_quotes'])
                df_temp.loc[:,f"Negation_LLM_Events_{col_suffix}_evidence_{analysis_type}"] = df_temp[f"LLM_Events_{col_suffix}_evidence_{analysis_type}"].apply(lambda x: x['negation'])
                df_temp.loc[:,f"Caused_By_LLM_Events_{col_suffix}_evidence_{analysis_type}"] = df_temp[f"LLM_Events_{col_suffix}_evidence_{analysis_type}"].apply(lambda x: x['caused_by'])
                df_temp.loc[:,f"Event_Time_LLM_Events_{col_suffix}_evidence_{analysis_type}"] = df_temp[f"LLM_Events_{col_suffix}_evidence_{analysis_type}"].apply(lambda x: x['event_time'])
                df_temp.loc[:,f"Order_LLM_Events_{col_suffix}_evidence_{analysis_type}"] = df_temp[f"LLM_Events_{col_suffix}_evidence_{analysis_type}"].apply(lambda x: x['orders'])
                df_temp.loc[:,f"Case_Attributes_LLM_Events_{col_suffix}_evidence_{analysis_type}"] = df_temp[f"LLM_Events_{col_suffix}_evidence_{analysis_type}"].apply(lambda x: x['case_attributes'])   
                df_temp.loc[:,f"Actor_LLM_Events_{col_suffix}_evidence_{analysis_type}"] = df_temp[f"LLM_Events_{col_suffix}_evidence_{analysis_type}"].apply(lambda x: x['actor'])   
                gt_file = f"../exports/04_groundtruth/P-SET/Annotated/{ET}_{focus_type}.pkl"
                if os.path.exists(gt_file):
                    gt_df = pd.read_pickle(gt_file) 
                    id_to_gt = {str(row[id_type]):row[gt_column] for _,row in gt_df.iterrows()}     
                    df_temp[gt_column] = df_temp[id_type].map(id_to_gt)  
                df_temp.to_excel(f"{output_folder}/{file_name}_att_{attribute_output}.xlsx", index=False)
                df_temp.to_pickle(f"{output_folder}/{file_name}_att_{attribute_output}.pkl")
                