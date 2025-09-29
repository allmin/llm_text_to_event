import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import subprocess
import ollama
import regex as re
import pandas as pd
import sys, os
from itertools import product
import time
from tqdm import tqdm


parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import config

class EventExtractor:
    def __init__(self, is_event_model_type=None, event_name_model_type=None, attribute_model_type=None, dictionary_file=None, llm_type=None):
        self.event_name_model_type = event_name_model_type
        self.event_name_known = False
        self.event_detection_time = []
        self.is_event_model_type = is_event_model_type
        self.attribute_model_type = attribute_model_type
        self.error_logs = []
        #dictionary denotes keyword matching, biolord denotes embedding similarity
        if event_name_model_type == "biolord":
            self.model = SentenceTransformer('FremyCompany/BioLORD-2023')
        elif event_name_model_type == "dictionary":
            if not dictionary_file:
                dictionary_file = "../resources/keyword_dict_annotated.xlsx"
            self.dictionary_input_df = pd.read_excel(dictionary_file)
            self.lemma_data = self.get_lemma_data() #['lemma', 'all_forms']
            self.dictionary_positive_lemmas = {lemma_keyword:event_type for keyword,event_type,positive in zip(self.dictionary_input_df['label'], self.dictionary_input_df['class'], self.dictionary_input_df['positive']) if positive==1 for lemma_keyword in self.lemmatize_keyword(keyword)}
            #update self.dictionary_positive_lemmas with all other forms of the key of the same lemma
            self.dictionary_positive_lemmas_list = [{'form':form,'event_type':event_type, 'lemma':lemma} for (lemma,event_type) in self.dictionary_positive_lemmas.items() for form in self.get_all_forms(lemma)]
            #write self.dictionary to a file "resources/keyword_dict_annotated_with_medication_expanded.xlsx"
            self.dictionary_expanded_df = pd.DataFrame(self.dictionary_positive_lemmas_list)
            #soring the dataframe by class
            self.dictionary_expanded_df.sort_values(by=['event_type','form'],ascending = True, inplace=True)
            self.dictionary_expanded_df.to_excel(dictionary_file.rstrip(".xlsx")+"_expanded.xlsx", index=False)
            self.dictionary = {form:{'event_type':event_type, 'lemma':lemma} for form,event_type,lemma in zip(self.dictionary_expanded_df['form'], self.dictionary_expanded_df['event_type'], self.dictionary_expanded_df['lemma'])}
        
        elif event_name_model_type == "llm":          
            if not llm_type:
                self.llm_type = "llama3.1:8b"  # Default model
            else:
                self.llm_type = llm_type
            
            # self.llm_type = "llama3.1:70b" --- IGNORE ---
            print(f"Using Ollama model: {self.llm_type}")
        
        
        self.is_event_cache = {}
        self.event_name_cache = {}
        self.event_id_cache = {}
        self.attributes_cache = {}
        self.keywords_cache = {}
        self.lemmas_cache = {}
        self.attribute_cache = {}
        self.event_attribute_cache = {}
        self.raw_output_cache = {}
        self.phrases_cache = {}
        self.positions_cache = {}
        self.text_quotes_cache = {}
        self.negation_cache = {}
        self.event_time_cache = {}
        self.caused_by_cache = {}
        self.orders_cache = {}
        self.case_attribute_cache = {}
        self.actor_cache = {}
    
    
    def lemmatize_keyword(self,keyword):
        lemma_dict = {}
        for sub_keyword in keyword.split("_"):
            returned_lemmas = self.lemma_data[self.lemma_data['all_forms'].apply(lambda x: sub_keyword in x)]['lemma'].tolist()
            if len(returned_lemmas) == 0:
                returned_lemmas = [sub_keyword]
            lemma_dict[sub_keyword] = returned_lemmas
        combinations = ['_'.join(p) for p in product(*lemma_dict.values())]
        return combinations
    
    def flatten_list_of_lists(self,list_of_lists):
        return [item for sublist in list_of_lists for item in sublist]
    
    def get_all_forms(self,lemma):
        all_form_dict = {}
        for sub_lemma in lemma.split("_"):
            returned_all_forms = self.lemma_data[self.lemma_data['lemma'] == sub_lemma]['all_forms'].tolist()
            returned_all_forms = self.flatten_list_of_lists(returned_all_forms)
            if len(returned_all_forms) == 0:
                 returned_all_forms = [sub_lemma]
            all_form_dict[sub_lemma] = returned_all_forms
        combinations = ['_'.join(p) for p in product(*all_form_dict.values())]
        return combinations
    
    def extract_events(self, texts, event_names, event_descriptions=None, threshold=0.2, 
                       attribute_output=False,
                       prompt_version=1,
                       prompt_evidence={'keywords':[],'event_names':[],'similarities':[]},
                       keyword_output=False, phrase_output=False, 
                       keyword_input=False, embedder_input=False,
                       example_input=False,
                       embedder_method=1):
        self.event_list = []
        self.prompt_version = prompt_version
        self.embedder_method=embedder_method
        self.texts = texts
        self.event_descriptions = event_descriptions
        self.similarities_dict = [{}]*len(self.texts)
        for attr in ["attributes",  "orders", "event_id", "text_quotes", "attribute_dict_list", "keywords",
                     "keyword_positions", "phrases", "raw_outputs", "lemmas", "event_name_prompt_list",
                     "event_time","negation","caused_by","case_attribute", "actor"]:
            setattr(self, attr, [""] * len(self.texts))
        self.predefined_event_names = event_names
        self.predefined_event_names_w_unknown = event_names + ["Unknown"]
        self.threshold = threshold
        self.keyword_output = keyword_output
        self.attribute_output = attribute_output
        self.phrase_output = phrase_output
        self.keyword_input = keyword_input
        self.embedder_input = embedder_input
        self.example_input = example_input
        self.prompt_evidence = prompt_evidence
        self.extract_is_event()
        self.extract_event_names()
        if self.attribute_model_type:
            self.extract_attributes()
        if len(self.event_detection_time) == 0:
            self.event_detection_time = [""]*len(self.texts)
            
        fields = ["texts", "predicted_events", "event_id", "similarities_dict", "keywords", "keyword_positions",
                  "attributes", "orders", "text_quotes", "phrases", "lemmas", "attribute_dict_list",
                  "event_name_prompt_list", "raw_outputs", "event_detection_time", "event_time", "negation", "caused_by","case_attribute", "actor"]

        # ✅ Check all fields are the same length
        lengths = [len(getattr(self, field)) for field in fields]
        if len(set(lengths)) != 1:
            debug_lengths = ", ".join(f"{field}={length}" for field, length in zip(fields, lengths))
            raise AssertionError(f"Inconsistent field lengths: {debug_lengths}")
        
        zipped_data = zip(*(getattr(self, field) for field in fields))
        
        for items in zipped_data:
            (text, event, event_id, similarity_dict, keyword, keyword_position,
             attribute, order, text_quote, phrase, lemma, attribute_dict,
             prompt, raw_output, detection_time, event_time, negation, caused_by, case_attribute, actor) = items
            if self.event_name_model_type == "biolord":
                self.event_list.append({"text":text, "event":event, "similarity":similarity_dict, "attributes": attribute_dict, "event_detection_time":detection_time})
            elif self.event_name_model_type == "dictionary":
                self.event_list.append({"text":text, "event":event, "keyword":keyword, "keyword_position":keyword_position, "lemma":lemma, "attributes": attribute_dict, "event_detection_time":detection_time})
            elif self.event_name_model_type == "llm":
                self.event_list.append({"text":text, "event":event,"event_id":event_id, "phrase":phrase, "raw_output":raw_output, 
                                        "attributes": attribute, "orders": order, "text_quotes": text_quote, 
                                        "event_time":event_time, "negation":negation, "caused_by":caused_by, "event_name_prompt":prompt, 
                                        "case_attributes":case_attribute, "actor":actor,
                                        "event_detection_time":detection_time})
            else:
                self.event_list.append({"text":text, "event":event, "attributes": attribute_dict, "event_detection_time":detection_time})
        return self.event_list
    
    
    def extract_attributes_given_event(self, text, event_name, skip_others=True):
        attributes_dict = {}
        if skip_others and event_name == "Unknown":
            return attributes_dict
        if self.attribute_model_type != "llm":
            return attributes_dict
        if text in self.event_name_cache:
            print("skipping LLM since text found in cache")
            return self.event_name_cache[text]
        prompt = f"""You are an expert medical language model that extracts structured data from clinical notes.

            Given a text that describes a patient-related {event_name} event, your task is to:
            1. Detect the relevant attribute types associated with that event (e.g., location, quality, duration, time, medication, dosage, etc.).
            2. Extract the corresponding attribute values from the text.

            Only extract attributes that are explicitly mentioned. Do not infer missing information.

            ### Input text:
            "{text}"

            ### Output Format:
            {{  "<attribute_type_1>": "<attribute_value_1>",
                "<attribute_type_2>": "<attribute_value_2>",
                ...}}

            If no attributes are found, return an empty "attributes" dictionary.

            Only output valid JSON. 

        """
        json_response, raw_output = self.get_json_response(prompt)

        try:
            attributes_dict = json.loads(json_response)
        except Exception as e:
            attributes_dict = {"ERROR":e, "input":json_response}
            self.error_logs.append(["is-event", text, attributes_dict ])
        self.event_name_cache[text] = attributes_dict
        return attributes_dict
    
    def extract_event_attributes_given_is_event(self, text, is_event):
        event_names = {}
        attributes_dict = {"event_name":"None", "attributes":{}}
        if is_event == "false":
            return attributes_dict
        if self.attribute_model_type != "llm":
            return attributes_dict
        if text in self.event_attribute_cache:
            return self.event_attribute_cache[text]
        prompt = f"""You are an expert medical language model that extracts structured data from clinical notes.
            Given a text that describes a patient-related event among {self.predefined_event_names}, your task is to:
            1. Detect the main event in the text among {self.predefined_event_names}. If nothing applies, set it as "Unknown".
            2. Detect the relevant attribute types associated with that main event (e.g., location, quality, duration, time, medication, dosage, etc.).
            3. Extract the corresponding attribute values from the text.
            Only extract attributes that are explicitly mentioned. Do not infer missing information.
            ### Input text:
            "{text}"

            ### Output Format:
            {{  "event_name": "<name_of_main_event>",
                "attributes":{{          
                "<attribute_type_1>": "<attribute_value_1>",
                "<attribute_type_2>": "<attribute_value_2>",
                ...}}}}

            If no attributes are found, return an empty "attributes" dictionary.
            ONLY output valid JSON. 

        """
        try:
            json_response,raw_output = self.get_json_response(prompt)
            attributes_dict = json.loads(json_response)
            if attributes_dict == {}:
                attributes_dict = {"event_name":"None", "attributes":{}}
        except Exception as e:
            self.error_logs.append(["event-attribute",text, json_response])
            attributes_dict={"event_name":"Error", "attributes":json_response}
        self.event_attribute_cache[text] = attributes_dict
        return attributes_dict

    
    
    def extract_is_event(self):
        if self.is_event_model_type == "llm":
            self.extract_is_event_llama()

    def extract_event_names(self):
        if self.event_name_model_type == "biolord":
            self.extract_event_names_biolord()
        if self.event_name_model_type == "llm":
            self.extract_event_names_llama()
        if self.event_name_model_type == "dictionary":
            self.dictionary = {form:values for (form,values) in self.dictionary.items() if values["event_type"] in self.predefined_event_names}
            self.extract_event_names_dictionary()
            
    
    def extract_event_names_dictionary(self):
        self.predicted_events = []
        self.event_detection_time = []
        self.keywords = []
        self.lemmas = []
        self.keyword_positions = []
        for ind,text in enumerate(self.texts):
            used_cache = False
            start_time = time.perf_counter()
            text_events = []
            text_keywords = []
            text_lemmas = []
            text_positions = []
            if text in self.event_name_cache:
                used_cache = True
                self.predicted_events.append(self.event_name_cache[text])
                self.keywords.append(self.keywords_cache[text])
                self.lemmas.append(self.lemmas_cache[text])
                self.keyword_positions.append(self.positions_cache[text])
            else:
                for index,(keyword, event_name_lemma_dict) in enumerate(self.dictionary.items()):
                    keyword_w_space = keyword.replace('_', ' ')
                    matches = list(re.finditer(rf'\b{re.escape(keyword_w_space)}\b', text, re.IGNORECASE))
                    if matches:
                        event_name = event_name_lemma_dict['event_type']
                        lemma = event_name_lemma_dict['lemma']
                        for m in matches:
                            text_events.append(event_name)
                            text_keywords.append(keyword_w_space)
                            text_lemmas.append(lemma)
                            text_positions.append(f"{m.span()[0]}_{m.span()[1]}")  # (start, end)
                            
                            
                            
                    # if re.search(rf'\b{re.escape(keyword_w_space)}\b', text, re.IGNORECASE):
                    #     num_of_occurrence = len(re.findall(rf'\b{re.escape(keyword_w_space)}\b', text, re.IGNORECASE))
                    #     event_name = event_name_lemma_dict['event_type']
                    #     lemma = event_name_lemma_dict['lemma']
                    #     for i in range(num_of_occurrence):
                    #         text_events.append(event_name)
                    #         text_keywords.append(keyword_w_space)
                    #         text_lemmas.append(lemma)
                if len(text_events)==0 and index==len(self.dictionary)-1:
                    text_events = ["Unknown"]
                    text_keywords = [""]
                    text_lemmas = [""]
                    text_positions = [""] 
                self.predicted_events.append(text_events)
                self.keywords.append(text_keywords)
                self.lemmas.append(text_lemmas)
                self.keyword_positions.append(text_positions)
                self.event_name_cache[text] = text_events
                self.keywords_cache[text] = text_keywords
                self.lemmas_cache[text] = text_lemmas
                self.positions_cache[text] = text_positions
            end_time = time.perf_counter()  # End timing
            elapsed_time = end_time - start_time
            if used_cache:
                #append nan
                self.event_detection_time.append(np.nan)
            else:
                self.event_detection_time.append(elapsed_time)
        self.event_name_known = True
            
    def extract_attributes(self):
        self.attribute_dict_list = []
        if self.event_name_known:
            for event_name, text in zip(self.predicted_events, self.texts):
                self.attribute_dict_list.append(self.extract_attributes_given_event(text,event_name))

        else:
            self.predicted_events = []
            for is_event, text in zip(self.is_events, self.texts):
                self.result = self.extract_event_attributes_given_is_event(text,is_event)
                if "attributes" not in self.result:
                    self.result['attributes'] = {}
                if "event_name" not in self.result:
                    self.result['event_name'] = "None"
                self.attribute_dict_list.append(self.result['attributes'])
                self.predicted_events.append(self.result['event_name'])

                
            

    def extract_event_names_biolord(self,use_faiss=False):
        self.predicted_events = []
        self.similarities = []
        start_time = time.perf_counter()
        
        self.text_embeddings = self.model.encode(self.texts, normalize_embeddings=True)
        self.label_embeddings = self.model.encode(self.predefined_event_names, normalize_embeddings=True)

        # FAISS expects float32
        query_vecs = np.array(self.text_embeddings).astype('float32')
        label_vecs = np.array(self.label_embeddings).astype('float32')
        
        if use_faiss:
            # Build FAISS index for cosine similarity (via inner product after normalization)
            dim = label_vecs.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # Inner Product = Cosine if vectors are normalized
            self.index.add(label_vecs)

            # Perform search: top-1 similar label per text
            self.similarities, self.indices = self.index.search(query_vecs, 1)
            self.predicted_events = self.get_event_names(self.threshold)
            self.event_name_known = True
        else:
            if self.embedder_method == 1:
                self.similarities = np.dot(query_vecs, label_vecs.T)
                self.indices = np.argmax(self.similarities, axis=1).reshape(-1, 1)
                self.similarities_dict = [{k.split(' : ')[0]: round(v, 2) for k, v in zip(self.predefined_event_names, row)}
                            for row in self.similarities]
                self.predicted_events = [
                    self.predefined_event_names[i[0]] if (i[0] != -1 and s[i[0]] > self.threshold) else "Unknown"
                    for i,s in zip(self.indices, self.similarities)
                ]
                self.event_name_known = True
            elif self.embedder_method != 1:
                self.label_embeddings = self.model.encode(self.predefined_event_names, normalize_embeddings=True)
                anti_label_vecs = np.array(self.anti_label_embeddings).astype('float32')
                self.anti_similarities = np.dot(query_vecs, anti_label_vecs.T)
                self.similarities = np.dot(query_vecs, label_vecs.T)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        time_per_text = elapsed_time/len(self.texts)
        self.predicted_events = [i.split(' : ')[0] for i in self.predicted_events]
        self.event_detection_time = [time_per_text]*len(self.texts)

    def get_event_names(self, threshold):
        indices = [
            [int(idx[0]),sim[0]] if sim[0] > threshold else [-1,sim]
            for idx, sim in zip(self.indices, self.similarities)
        ]
        events = [[self.predefined_event_names[i],sim] if i != -1 else ["Unknown",sim] for i,sim in indices]
        return events

    def get_json_response(self, prompt):
        # athene-v2:72b
        response = ollama.generate(model=self.llm_type, prompt=prompt, options={"temperature": 0}, format='json')
        raw_output = response['response'].strip()
        # json_response = re.search(r'\{.*?\}', raw_output)
        json_response = re.search(r'\{(?:[^{}]|(?R))*\}', raw_output, re.DOTALL)
        # json_response = re.search(r'```(?:json)?\s*({.*?})\s*```', raw_output, re.DOTALL)
        if json_response: 
            try:      
                return json_response.group(0), raw_output
            except:
                return "{}",raw_output
        else:
            json_response="{}"
            return json_response,raw_output
    
    
    def extract_is_event_llama(self):
        self.is_events = []
        for text in self.texts:
            if text in self.is_event_cache:
                self.is_events.append(self.is_event_cache[text])
            else:
                prompt = f"""Does this text describe an activity related to any of {self.predefined_event_names}?. 
                Output ONLY a JSON: {{"is_activity":<boolean>}}
                text: {text}"""
                json_response, raw_output = self.get_json_response(prompt)
                try:
                    json_dict = json.loads(json_response)
                    is_event = json_dict.get("is_activity", False)
                    if (type(is_event) == str):
                        if is_event.lower() in ['true']:
                            is_event = True
                        else:
                            is_event = False
                except json.JSONDecodeError:
                    is_event = False
                self.is_event_cache[text]=is_event
                self.is_events.append(is_event)
    
    
    def extract_event_names_llama(self):
        self.event_detection_time = []
        self.predicted_events = []
        self.event_id = []
        self.event_name_prompt_list = []
        self.raw_outputs = []
        self.attributes = []
        self.text_quotes = []
        self.orders = []
        self.event_time = []
        self.caused_by = []
        self.negation = []
        self.case_attribute = []
        self.actor = []
        for ind, text in tqdm(enumerate(self.texts)):
            used_cache = False
            start_time = time.perf_counter()
            if self.event_descriptions:
                event_w_description = self.event_descriptions
                event_w_description["Unknown"] = """choose "Unknown" if none of the other event type are applicable."""
            else:
                event_w_description = self.predefined_event_names_w_unknown
            event_w_description = "\n".join([f"{k} : {v}" for (k,v) in event_w_description.items()]) if type(event_w_description)==dict else event_w_description
            detected_keywords = self.prompt_evidence['keywords'][ind] if ind < len(self.prompt_evidence['keywords']) else []
            dct = self.prompt_evidence['dct'][ind] if ind < len(self.prompt_evidence['dct']) else []
            prompt = config.get_general_prompt_template(text=text, 
                                                        predefined_event_names=self.predefined_event_names_w_unknown, 
                                                        prompt_version=self.prompt_version, 
                                                        event_w_description=event_w_description, 
                                                        attribute_output=self.attribute_output, 
                                                        keyword_input=self.keyword_input, 
                                                        example_input=self.example_input, 
                                                        detected_keywords=detected_keywords,
                                                        dct=dct)
            if text in self.event_name_cache:
                used_cache = True
                self.predicted_events.append(self.event_name_cache[text])
                self.event_id.append(self.event_id_cache[text])
                self.attributes.append(self.attributes_cache[text])
                self.text_quotes.append(self.text_quotes_cache[text])
                self.raw_outputs.append(self.raw_output_cache[text])
                self.orders.append(self.orders_cache[text])
                self.negation.append(self.negation_cache[text])
                self.caused_by.append(self.caused_by_cache[text])
                self.event_time.append(self.event_time_cache[text])
                self.case_attribute.append(self.case_attribute_cache[text])
                self.actor.append(self.actor[text])
            else:                
                self.json_response, raw_output = self.get_json_response(prompt)
                if self.json_response:
                    try:
                        event = json.loads(self.json_response)
                        print(text,event)
                        event_id = []
                        event_name = []
                        attributes = []
                        text_quotes = []
                        negation = []
                        caused_by = []
                        event_time = []
                        orders = []
                        case_attribute = []
                        actor = []
                        for event_inst in event.get('events', []):
                            event_id.append(event_inst.get("event_id","Unknown"))
                            event_name.append(event_inst.get("event_type","Unknown"))
                            text_quotes.append(event_inst.get("text_quote","Unknown"))
                            negation.append(event_inst.get("negation", "Unknown"))
                            caused_by.append(event_inst.get("caused_by", "Unknown"))
                            event_time.append(event_inst.get("time", "Unknown"))
                            actor.append(event_inst.get("actor", "Unknown"))
                            attributes.append({event_inst.get("event_type","Unknown"):event_inst.get("event_attributes",{})}) 
                        orders.append(event.get("order","[]"))
                        case_attribute.append(event.get("case_attributes",{}))
                    except Exception as e:
                        print(f"Exception: {e}, Index: {ind}, Text: {text}, JSON Response: {self.json_response}")
                assert len(event_name) == len(text_quotes) == len(attributes), f"{len(event_name)}, {len(text_quotes)}, {len(attributes)}"
                self.predicted_events.append(event_name)
                self.event_id.append(event_id)
                self.attributes.append(attributes)
                self.raw_outputs.append(raw_output)
                self.text_quotes.append(text_quotes)
                self.negation.append(negation)
                self.caused_by.append(caused_by)
                self.event_time.append(event_time)
                self.orders.append(orders)
                self.case_attribute.append(case_attribute)
                self.actor.append(actor)
                self.event_name_cache[text]=event_name
                self.event_id_cache[text]=event_id
                self.attributes_cache[text]=attributes
                self.raw_output_cache[text]=raw_output
                self.text_quotes_cache[text]=text_quotes
                self.negation_cache[text]=negation
                self.caused_by_cache[text]=caused_by
                self.event_time_cache[text]=event_time
                self.orders_cache[text]=orders
                self.case_attribute_cache[text]=case_attribute
                self.actor_cache[text]=actor
                
            self.event_name_prompt_list.append(prompt)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            if used_cache:
                #append nan
                self.event_detection_time.append(np.nan)
            else:
                self.event_detection_time.append(elapsed_time)
        self.event_name_known = True
    
   
    def get_lemma_data(self):
        with open("../resources/lemma.en.txt", "r", encoding="utf-8") as file:
            lines = file.readlines()
        lemma_data = []
        for line in lines:
            lemma_part, forms_part = line.strip().split(" -> ")
            lemma = lemma_part.split("/")[0]  # Remove frequency
            all_forms = forms_part.split(",") + [lemma]
            lemma_data.append({"lemma": lemma, "all_forms": all_forms})
        lemma_data = pd.DataFrame(lemma_data)
        return lemma_data
    
if __name__ == "__main__":
    mtexts = ["He slept well","He slept well","He began to break bread","Trigonometry"]
    mtexts = ["at midnight after being asleep in bed for several hours patient awoke coughing unable to raise unless sitting up in chair",
                  "labile bp, apnea ventilation when asleep even when off sedation, vent mode changed to mmv by rt",
                  "patient restless and agitated @ times, pulling cpap off, more cooperative when medicated, short periods of apnea noted when off cpap, will attempt to keep patient on cpap when asleep",
                  "asleep ft on aggressive pulm toilet done cpt with effect bs controlled with insulin drops and prn boluses [**md number(3) **] output great lasix tid cont to monitor cr and renal fx"]
    mtexts = ["""had difficulty falling asleep but refused serax. slept very well. became disoriented from 3:00-5:00, knew he was still in hospital but asked about, "the dog in the room."""]
    mtexts = ["""CCU NSG ADMIT NOTE.\n19 year old FEMALE ADMITTED TO CCU status post VF ARREST.\n\nPMH:NOT SIGNIFICANT.\n\nALLERGIES:NKDA.\n\nMEDS:MULTIPLE OVER THE COUNTER DIET SUPRESSENTS.\n\nhistory:?VIRAL SYNDROME APPROX 2 WEEKS AGO. TAKING DIET SURPRESSENTS- ?ONSET OF USE (PER FAMILY PLANNING [**State 2968**] VACATION OVER HOLIDAY-?ING ONSET OF DIET SURPRESSENTS). [**1-8**] AM ONSET ACUTE DYSPNEA W PROGRESSION TO CARDIAC ARREST-VF. INTUBATED & DEFIB TO ST IN FIELD-TRANSPORTED TO [**Hospital1 2**]. AGGRESSIVELY RXED IN EW. CT HEAD/CHEST-NEG FOR INTRACRANIAL HEMORRHAGE & PE. FEBRILE-PAN CULTURED & ABX STARTED. PROGRESSIVELY DETERIORATED- REQUIRING PRESSORS & APPROX 4L FL-TO CARD CATH LAD-CLEAN C'S, BUT ELEVATED FILLING PRESSURES-W 30'S-RX W LASIX & ADMITTED TO CCU FOR FURTHER MANAGEMENT.\n   ECHO=SEVER GLOBAL LV HK. LV FUNCTION SEVERLY DEPRESSED. RV FUNCTION DEPRESSED. 1+MR.\n\nSOCIAL:BU STUDENT-2ND YEAR. FROM ILL. PARENTS CONTACT[**Name (NI) **] & BOTH PRESENT. HAS 2 OTHER SIBLINGS IN ILL. NON SMOKER & ?LIMITED DRINKER.\n\n\n"""]
    mtexts = ["The patient slept in the morning, took a nap in the afternoon, and had a good night's sleep."]
    mtexts = ["The patient couldn't eat due to severe pain in the throat.", "Due to the constipation, the patient couldnt sleep well and was in constant pain thought the day"]
    mtexts = ["patient reported history of sleep apnea and uses oral airway piece at hs at home"]
    
#     mtexts = ['bp lower when asleep',
#  'sleeping in naps',
#  'slept well',
#  'sleeping in long naps',
#  'short naps',
#  'slept most of night',
#  'slept in naps',
#  'sleeping',
#  'slept most of noc',
#  'slept most of night']
    mevent_names = ["Eating","Excretion","Family","Pain", "Sleep"]
    event_names_w_descriptions = ["Eating : To take food into the body by mouth",
                                  "Exretion : The process by which waste matter is discharged from the body",
                                  "Pain : The sensation of discomfort, distress, or agony, resulting from the stimulation of specialized nerve endings.",
                                  "Sleep : A natural and periodic state of rest during which consciousness of the world is suspended."]

    DICT = EventExtractor(event_name_model_type='dictionary', attribute_model_type='None')
    DICT.extract_events(texts=mtexts, event_names=mevent_names)
    print("Dictionary_events:",DICT.event_list)


    
    LLAMA2 = EventExtractor(event_name_model_type='llm', attribute_model_type='None',llm_type = "llama3.1:70b")   
    LLAMA2.extract_events(texts=mtexts, event_names=mevent_names, 
                                event_descriptions=config.event_descriptions,
                                prompt_version=3,
                                prompt_evidence={'keywords':DICT.keywords, 
                                                 'event_names':DICT.predicted_events, 
                                                 'dct':[(23,45)]
                                                },
                                attribute_output=True, 
                                keyword_input=True, example_input=True,)
    print("LLAMA_all_evidence_events:",
          LLAMA2.event_list[0]["event_name_prompt"],
          LLAMA2.event_list[0]["attributes"], 
          LLAMA2.event_list[0]["text_quotes"],
          LLAMA2.orders)
    # sudo kill -9 $(nvidia-smi | awk 'NR>8 {print $5}' | grep -E '^[0-9]+$')
    #  srun --partition=gpu_h100 --gres=gpu:2 --cpus-per-task=18 --mem=100G --time=8:00:00 --pty bash -i
    


