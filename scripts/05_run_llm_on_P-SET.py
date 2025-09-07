import pandas as pd
import os, sys
from datetime import datetime
from itertools import product
from glob import glob

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from utils.event_extractor import EventExtractor
from config import event_types, event_description_dict_llm, llm_type

def extract_events_funct(texts, extractor=None, evidence={'keywords':[],'event_names':[],'similarities':[]}, keyword_input=None, example_input=None, attribute_output=None):
    global event_description_dict_llm, event_types
    events = extractor.extract_events(texts=texts, event_names=event_types, event_descriptions=event_description_dict_llm, threshold=0.2, prompt_evidence=evidence, keyword_input=keyword_input, example_input=example_input, attribute_output=attribute_output)
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



for ET in ['Sleep','Excretion','Eating','Family','Pain']:    
    for analysis_type in ['sentence', 'document']:
        for attribute_output in [True, False]:
        
            os.makedirs(f"../exports/llm_{llm_type}/{ET}", exist_ok=True)
            try:
                file = glob(f"../exports/groundtruth/P-SET/Generated/{ET}*.pkl")[0]
            except:
                print(f"No file found for {ET}")
                continue
            file_name = os.path.basename(file).strip(".pkl")
            df = pd.read_pickle(file)
            df.Similarity = df.Similarity.astype(str)
            df_grouped = df.groupby(['Sentence_dictionary'])[["UID","Event_Name_dictionary","Keyword","Similarity"]].agg(lambda x: tuple(set(x)) if len(set(x))>1 else set(x).pop()).reset_index()
            df_grouped.Similarity = df_grouped.Similarity.apply(eval)     
            disagreement_df_temp = df_grouped.copy()
            if analysis_type == 'sentence':
                input_to_analyse = disagreement_df_temp.Sentence_dictionary.tolist()
            elif analysis_type == 'document':
                input_to_analyse = disagreement_df_temp.DOCUMENT.tolist()
            print(ET,len(disagreement_df_temp), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), file_name)
            print(f"attribute_output:{attribute_output}, Time Start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
            evidence={'keywords':disagreement_df_temp.Keyword.tolist(), 'event_names':disagreement_df_temp.Event_Name_dictionary.tolist(), 'similarities':disagreement_df_temp.Similarity.tolist()}
            for keyword_input, example_input in [i for i in product([True,False],[True,False])]:
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
                disagreement_df_temp.loc[:,f"Attribute_LLM_Events_{col_suffix}_evidence_{analysis_type}"] = disagreement_df_temp[f"LLM_Events_{col_suffix}_evidence_{analysis_type}"].apply(lambda x: x['attribute'])
                disagreement_df_temp.to_excel(f"../exports/llm_{llm_type}/{ET}/{file_name}_att_{attribute_output}.xlsx", index=False)
                disagreement_df_temp.to_pickle(f"../exports/llm_{llm_type}/{ET}/{file_name}_att_{attribute_output}.pkl")
                
