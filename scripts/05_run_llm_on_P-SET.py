# CUDA_VISIBLE_DEVICES=0 python 05_run_llm_on_P-SET.py --attribute_output True
# CUDA_VISIBLE_DEVICES=1 python 05_run_llm_on_P-SET.py --attribute_output False
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
from config import event_types, event_description_dict_llm, llm_type, event_attributes_dict_llm, examples, examples_Ao
import argparse

def extract_events_funct(texts, extractor=None, evidence={'keywords':[],'event_names':[],'similarities':[]}, keyword_input=None, example_input=None, attribute_output=None):
    global event_description_dict_llm, event_types, event_attributes_dict_llm, examples, examples_Ao
    events = extractor.extract_events(texts=texts, 
                                      event_names=event_types, 
                                      event_descriptions=event_description_dict_llm, 
                                      prompt_evidence=evidence, 
                                      examples=examples_Ao if attribute_output else examples,
                                      attribute_description_dict=event_attributes_dict_llm,
                                      attribute_output=attribute_output,
                                      keyword_input=keyword_input, 
                                      example_input=example_input,
                                      )
    return events

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
parser.add_argument('--attribute_output', type=bool, default=False, help='Set attribute_output True or False')

args = parser.parse_args()

attribute_output_raw = args.attribute_output
print(f'attribute_output:{attribute_output_raw}, type:{type(attribute_output_raw)}')
dataset = 'P-SET'

for ET in ['Sleep','Excretion','Eating','Family','Pain'][:1]:    
    for attribute_output in [attribute_output_raw]:
        os.makedirs(f"../exports/05_llm_{llm_type}_{dataset}/{ET}", exist_ok=True)
        for analysis_type in ['Sent', 'Doc']:
            try:
                file = glob(f"../exports/04_groundtruth/{dataset}/Generated/{ET}*{analysis_type}*.pkl")[0]
            except:
                print(f"No file found for {ET}")
                continue
            file_name = os.path.basename(file).strip(".pkl")
            df = pd.read_pickle(file)
            df['Event_Name'] = [tuple(i) for i in df['Event_Name']]
            df['Keyword'] = [tuple(i) for i in df['Keyword']]
            
        
            if analysis_type == 'Sent':
                disagreement_df_temp = df.copy()
                input_to_analyse = disagreement_df_temp.Sentence.tolist()
            elif analysis_type == 'Doc':
                disagreement_df_temp = df.groupby('UID')[["Event_Name","Keyword","Document"]].agg(lambda x:combine_lists(x)).reset_index()
                disagreement_df_temp['Document'] = [i[0] for i in disagreement_df_temp['Document']]
                input_to_analyse = disagreement_df_temp.Document.tolist()
            disagreement_df_temp = disagreement_df_temp.copy().iloc[:300]
            print(ET,len(disagreement_df_temp), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), file_name)
            print(f'attribute_output:{attribute_output}, Time Start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            evidence={'keywords':disagreement_df_temp.Keyword.tolist(), 'event_names':disagreement_df_temp.Event_Name.tolist(), }
            for keyword_input, example_input in [i for i in product([True,False],[True,False])]:
                print(f"keyword_input:{keyword_input}, example_input:{example_input}, attribute_output:{attribute_output}, analysis_type:{analysis_type}")
                col_suffix = get_col_suffix(keyword_input, example_input)
                disagreement_df_temp.loc[:,f"LLM_Events_{col_suffix}_evidence_{analysis_type}"] = extract_events_funct(texts=input_to_analyse,
                                                                                                    extractor=EventExtractor(
                                                                                                        event_name_model_type="llm",
                                                                                                        attribute_model_type="None",
                                                                                                        llm_type=llm_type,
                                                                                                        ),
                                                                                                    keyword_input=keyword_input,
                                                                                                    attribute_output=attribute_output,
                                                                                                    evidence=evidence)
                disagreement_df_temp.loc[:,f"Event_Name_LLM_Events_{col_suffix}_evidence_{analysis_type}"] = disagreement_df_temp[f"LLM_Events_{col_suffix}_evidence_{analysis_type}"].apply(lambda x: x['event'])
                disagreement_df_temp.loc[:,f"Attribute_LLM_Events_{col_suffix}_evidence_{analysis_type}"] = disagreement_df_temp[f"LLM_Events_{col_suffix}_evidence_{analysis_type}"].apply(lambda x: x['attributes'])
                disagreement_df_temp.loc[:,f"Text_Quotes_LLM_Events_{col_suffix}_evidence_{analysis_type}"] = disagreement_df_temp[f"LLM_Events_{col_suffix}_evidence_{analysis_type}"].apply(lambda x: x['text_quotes'])
                disagreement_df_temp.to_excel(f"../exports/05_llm_{llm_type}_{dataset}/{ET}/{file_name}_att_{attribute_output}.xlsx", index=False)
                disagreement_df_temp.to_pickle(f"../exports/05_llm_{llm_type}_{dataset}/{ET}/{file_name}_att_{attribute_output}.pkl")
                
