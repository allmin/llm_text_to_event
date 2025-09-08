import pandas as pd
import os, sys
from datetime import datetime
from itertools import product
from glob import glob

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from utils.event_extractor import EventExtractor
from config import event_types, event_description_dict_llm, llm_type, event_attributes_dict_llm, examples

def extract_events_funct(texts, extractor=None, evidence={'keywords':[],'event_names':[],'similarities':[]}, keyword_input=None, example_input=None, attribute_output=None):
    global event_description_dict_llm, event_types, event_attributes_dict_llm, examples
    events = extractor.extract_events(texts=texts, 
                                      event_names=event_types, 
                                      event_descriptions=event_description_dict_llm, 
                                      prompt_evidence=evidence, 
                                      examples=examples,
                                      attribute_description_dict=event_attributes_dict_llm,
                                      attribute_output=attribute_output,
                                      keyword_input=keyword_input, 
                                      example_input=example_input,
                                      )
                              #    LLAMA2.extract_events(texts=mtexts, event_names=mevent_names, 
                                                                #                     event_descriptions=event_description_dict_llm,
                                                                #                     prompt_evidence={'keywords':DICT.keywords, 
                                                                #                                      'event_names':DICT.predicted_events, 
                                                                #                                      'similarities':BIOLORD.similarities_dict},
                                                                #                     examples=examples,
                                                                #                     attribute_description_dict=event_attributes_dict_llm,
                                                                #                     attribute_output=True, 
                                                                #                     keyword_input=True, example_input=True,)
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


dataset="P-SET"
for ET in ['Sleep','Excretion','Eating','Family','Pain'][:1]:    
    for attribute_output in [True, False]:
        os.makedirs(f"../exports/llm_{llm_type}_{dataset}/{ET}", exist_ok=True)
        try:
            file = glob(f"../exports/groundtruth/{dataset}/Generated/{ET}*.pkl")[0]
        except:
            print(f"No file found for {ET}")
            continue
        file_name = os.path.basename(file).strip(".pkl")
        df = pd.read_pickle(file)
        df['Event_Name'] = [tuple(i) for i in df['Event_Name']]
        df['Keyword'] = [tuple(i) for i in df['Keyword']]
        # df.Similarity = df.Similarity.astype(str)
        # df_grouped = df.groupby(['Sentence'])[["UID","Event_Name","Keyword","DOCUMENT"]].agg(lambda x: tuple(set(x)) if len(set(x))>1 else set(x).pop()).reset_index()
        # df_grouped.Similarity = df_grouped.Similarity.apply(eval)     
        # disagreement_df_temp = df_grouped.copy()
        disagreement_df_temp = df.copy()
        for analysis_type in ['sentence', 'document']:
            if analysis_type == 'sentence':
                input_to_analyse = disagreement_df_temp.Sentence.tolist()
            elif analysis_type == 'document':
                input_to_analyse = disagreement_df_temp.DOCUMENT.tolist()
            print(ET,len(disagreement_df_temp), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), file_name)
            print(f"attribute_output:{attribute_output}, Time Start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
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
                disagreement_df_temp.to_excel(f"../exports/llm_{llm_type}_{dataset}/{ET}/{file_name}_att_{attribute_output}.xlsx", index=False)
                disagreement_df_temp.to_pickle(f"../exports/llm_{llm_type}_{dataset}/{ET}/{file_name}_att_{attribute_output}.pkl")
                
