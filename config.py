llm_type = "llama3.1:70b"
event_types = ["Pain", "Sleep", "Excretion", "Eating", "Family"]
# event_descriptions = {"Eating": "The patient takes food into their body by mouth. Identifed Always",
#                        "Excretion": "The patient discharges waste matter from their body. Identifed Always",
#                        "Family": "The patient has a visit, call, or communication with a family member.",
#                        "Pain": "The patient reports or shows signs of pain. Identifed Always",
#                        "Sleep": "The patient sleeps or the sleep’s quality or quantity is described. Identifed Always."
#                             }
event_descriptions = {"Eating": "The patient takes food into their body by mouth.",
                       "Excretion": "The patient discharges waste matter from their body.",
                       "Family": "The patient has a visit, call, or communication with a family member.",
                       "Pain": "The patient reports or shows signs of pain.",
                       "Sleep": "The patient sleeps or the sleep’s quality or quantity is described."
                            }
event_description_dict_embedder = event_descriptions



def get_general_prompt_template(text, predefined_event_names, event_w_description, prompt_version, attribute_output, keyword_input, example_input, detected_keywords, dct):
    task_description = get_task_description(attribute_output,keyword_input)
    classification_rules = get_classification_rules(attribute_output,prompt_version)
    event_attributes_dict_llm = get_event_attributes_dict()
    attribute_description = get_attribute_clause(event_attributes_dict_llm)
    keyword_evidence = get_keyword_clause(detected_keywords)
    output_format = get_output_format(predefined_event_names, attribute_output, prompt_version)
    output_rules = get_output_rules(attribute_output,prompt_version)
    examples = get_examples(attribute_output)
    if prompt_version == 1:
        general_prompt_template = f"""Given the text: {text}, 
                                and the following event types with their definitions: {event_w_description} 
                                ---
                                Your task:
                                {task_description}
                                ---
                                Classification Rules:
                                {classification_rules}
                                {attribute_description if attribute_output else ""}
                                {keyword_evidence if keyword_input else ""}
                                ---
                                Output Schema (strict):
                                {output_format}
                                {output_rules}
                                {examples if example_input else ""}
                                """
    elif prompt_version == 2:
        general_prompt_template = f"""
        **Classification and Attribute Extraction Task** 
            Classify the following sentence into events that took place DURING THE SHIFT in which this note was written, 
            using one or more of the following categories: {predefined_event_names}. 
            {classification_rules}
            For each detected event, output strictly valid JSON following the schema below: 
            ```
           {output_format}  
           ```
           {output_rules}
           The text:
           {text}
        """
    elif prompt_version == 3:
        general_prompt_template = f"""
        **Classification and Attribute Extraction Task** 
            Classify the following sentence into events that took place DURING THE SHIFT in which this note was written, 
            using one or more of the following categories: {event_w_description}. 
            {classification_rules}
            For each detected event, output strictly valid JSON following the schema below: 
            ```
           {output_format}  
           ```
           {output_rules}
           ---
           Text written between: {dct[0]} and {dct[1]}
           Text:
           {text}
        """
        
    elif prompt_version == 4:
        general_prompt_template = f"""
        **Classification and Attribute Extraction Task** 
            Classify the following sentence into events that took place DURING THE SHIFT in which this note was written, 
            using one or more of the following categories: 
            {event_w_description}.
            
            Conditions: 
            {classification_rules}
            For each detected event, output strictly valid JSON following the schema below: 
            ```
              {output_format}  
            ```
           {output_rules}
           
           Text (written between {dct[0]} and {dct[1]}):
           {text}
        """
    elif prompt_version == 5:
        general_prompt_template = f"""
        **Classification and Attribute Extraction Task** 
            Classify the Text at the end of this prompt into events that took place at the HOSPITAL, DURING THE SHIFT in which this text was written, 
            using one or more of the following categories: 
            {event_w_description}.
            
            Conditions: 
            {classification_rules}
            For each detected event, output strictly valid JSON following the schema below: 
            ```
              {output_format}  
            ```
           {output_rules}
           
           Text (written between {dct[0]} and {dct[1]}):
           {text}
        """
        
        
    return general_prompt_template



    

def get_event_attributes_dict():
    event_attributes_dict_llm = {
                                    "Eating": {
                                        "food": "<string> # e.g., pancakes, porridge, etc.",
                                        "amount": "<string> # e.g., 1, 2 bowls, etc.",
                                        "method": "<string>  # e.g., oral, tube, etc."
                                    },
                                    "Excretion": {
                                        "type": "<string> # e.g., urine, stool, etc.",
                                        "frequency": "<string> # e.g., 2x, once, etc.",
                                        "quality": "<string> # e.g., loose, hard, etc."
                                    },
                                    "Family": {
                                        "interaction": "<string> # e.g., visit, call, communication, etc.",  
                                        "relation": "<string> # e.g., mother, son, etc."
                                    },
                                    "Pain": {
                                        "severity": "<string> # e.g., mild, moderate, severe, numeric scale if present, etc.",
                                        "location": "<string> # e.g., right knee, head, etc.",
                                        "duration": "<string> # e.g., all night, 3 hours, etc.",
                                    },
                                    "Sleep": {
                                        "quality": "<string> # e.g., good, poor, restless, etc.",   
                                        "duration": "<string> # e.g., 6 hours, phrases like \"all night\", etc."   
                                    }
                                }
    common_attributes = {"negation": "<boolean> # true if the event is negated, false otherwise",
                        "time": "<string> # e.g., 2 am, 5 pm, night, evening etc.",
                        "caused_by": "<string> # e.g., medication, treatment, etc.",
                        }

    for et in event_types:
        event_attributes_dict_llm[et].update(common_attributes)
    
    return event_attributes_dict_llm 
    
def get_attribute_clause(attribute_description_dict):
    f"""---
        Attributes allowed:
        {attribute_description_dict}
        If no relevant attribute exists for an event, return an empty object {{}}.
        """

def get_keyword_clause(detected_keywords):
    return f"""---
        Keyword evidence (Ki):
        Additional facts: A keyword matching algorithm without context detected keyword(s) {detected_keywords}.
        Use this only as a pointer. If context contradicts, ignore."""

def get_examples(attribute_output):
    examples_base = """
                ---
                Examples:
                
                text: "Patient ate breakfast this morning. He seems less anxious."
                output: {
                "events": [
                    {"event_id": "e1",
                    "event_type": "Eating",
                    "text_quote":"Patient ate breakfast this morning"
                    }
                ],
                "order":[]
                }
                
                text: "Patient reported severe abdominal pain."
                output: {
                "events": [
                    {"event_id": "e1",
                    "event_type": "Pain",
                    "text_quote": "severe abdominal Pain"
                    }
                ],
                "order":[]
                }
                
                text: "Patient called his son around 3 pm."
                output: {
                "events": [
                    {"event_id": "e1",
                    "event_type": "Family",
                    "text_quote": "called his son around 3 pm"
                    }
                ],
                "order":[]
                }
                
                text: "Patient had a loose stool overnight."
                output: {
                "events": [
                    {"event_id": "e1",
                    "event_type": "Excretion",
                    "text_quote": "loose stool overnight"
                    }
                ],
                "order":[]
                }
                
                text: "Patient was able to sleep well last night."
                output: {
                "events": [
                    {"event_id": "e1",
                    "event_type": "Sleep",
                    "text_quote": "sleep well last night"
                    }
                ],
                "order":[]
                }
                
                text: "The patient couldn\'t sleep due to severe pain."
                output: {
                "events": [
                    {"event_id": "e1",
                    "event_type": "Pain",
                    "text_quote": "severe pain"
                    }
                ],
                "order":[]
                }
                
                text: "The patient complained of severe back pain, was given Tylenol, but the pain persisted and he was then prescribed stronger morphine."
                output: {
                "events": [
                    {"event_id": "e1",
                    "event_type": "Pain",
                    "text_quote": "complained of severe back pain"
                    },
                    {"event_id": "e2",
                    "event_type": "Pain",
                    "text_quote": "pain persisted"
                    }
                ],
                "order":[]
                }
                """
    examples_Ao = """
                ---
                Examples:
                
                text: "Patient ate breakfast this morning. He seems less anxious."
                output: {
                "events": [
                    {"event_id": "e1",
                    "event_type": "Eating",
                    "text_quote":"Patient ate breakfast this morning",
                    "attributes": {"food": "breakfast", 
                                   "amount": "Unknown",
                                   "method": "Unknown",
                                   "negation": "false",
                                   "time":"morning",
                                   "caused_by":"Unknown"}
                    }
                ],
                "order":[]
                }
                
                text: "Patient reported severe abdominal pain."
                output: {
                "events": [
                    {"event_id": "e1",
                    "event_type": "Pain",
                    "text_quote": "severe abdominal Pain",
                    "attributes": {"severity": "severe", 
                                   "location": "abdominal",
                                   "duration": "Unknown",
                                   "negation": "false",
                                   "time":"Unknown",
                                   "caused_by":"Unknown"}
                    }
                ],
                "order":[]
                }
                
                text: "Patient called his son around 3 pm."
                output: {
                "events": [
                    {"event_id": "e1",
                    "event_type": "Family",
                    "text_quote": "called his son around 3 pm",
                    "attributes": {"interaction": "call", 
                                    "relation": "son",
                                    "negation": "false",
                                    "time":"3 pm",
                                    "caused_by":"Unknown"}
                    }
                ],
                "order":[]
                }
                
                text: "Patient had a loose stool overnight."
                output: {
                "events": [
                    {"event_id": "e1",
                    "event_type": "Excretion",
                    "text_quote": "loose stool overnight",
                    "attributes": {"type": "stool", 
                                   "quality": "loose", 
                                   "frequency":"overnight", 
                                   "negation": "false",
                                   "time":"night",
                                   "caused_by":"Unknown"}
                    }
                ],
                "order":[]
                }
                
                text: "Patient was able to sleep well last night."
                output: {
                "events": [
                    {"event_id": "e1",
                    "event_type": "Sleep",
                    "text_quote": "sleep well last night",
                    "attributes": {"quality": "well",
                                    "duration": "Unknown", 
                                    "negation": "false",
                                    "time":"night",
                                    "caused_by":"Unknown"}
                    }
                ],
                "order":[]
                }
                
                text: "The patient couldn\'t sleep due to severe pain."
                output: {
                "events": [
                    {"event_id": "e1",
                    "event_type": "Sleep",
                    "text_quote": "couldn\'t sleep",
                    "attributes": {"quality": "poor", 
                                    "duration": "Unknown",
                                    "negation": "true",
                                    "time":"Unknown",
                                    "caused_by":"Pain"}
                    },
                    {"event_id": "e2",
                    "event_type": "Pain",
                    "text_quote": "severe pain",
                    "attributes": {"severity": "severe", 
                                    "location": "Unknown", 
                                    "duration": "Unknown", 
                                    "time":"Unknown",
                                    "caused_by":"Unknown"}
                    }
                ],
                "order":["e1", "after", "e2"]
                }
                
                text: "The patient complained of severe back pain, was given Tylenol, but the pain persisted and he was then prescribed stronger morphine."
                output: {
                "events": [
                    {"event_id": "e1",
                    "event_type": "Pain",
                    "text_quote": "complained of severe back pain",
                    "attributes": {"severity": "severe", 
                                    "location": "back",
                                    "duration": "Unknown", 
                                    "time":"Unknown",
                                    "caused_by":"Unknown"}
                    },
                    {"event_id": "e2",
                    "event_type": "Pain",
                    "text_quote": "pain persisted",
                    "attributes": {"severity": "severe", 
                                   "location": "back",
                                   "duration": "persistent", 
                                   "time":"after Tylenol",
                                   "caused_by":"Unknown"}
                    }
                ],
                "order":[["e1", "before", "e2"]]
                }
                """
    if attribute_output:
        return examples_Ao
    else:
        return examples_base

def get_task_description(attribute_output, keyword_input):
    task_description = []
    task_description.append("Identify ALL events in the text (zero, one, or more).")
    task_description.append("For each event, assign an event_type from the list above.")
    task_description.append("If multiple event of same type is mentioned, assign the event_type multiple times")
    if attribute_output:
        task_description.append("Extract attributes for each event using ONLY the allowed attributes listed below.")
    if keyword_input:
        task_description.append("Use keyword evidence (Ki) ONLY if it is consistent with the text context.")
    task_description.append("Output strictly valid JSON following the schema below.")
    task_description = "\n".join([f"{i+1}. {task}" for i,task in enumerate(task_description)])
    return task_description
                
def get_classification_rules(attribute_output, prompt_version):
    classification_rules = []
    if prompt_version == 1:
        classification_rules.append("- A text may contain multiple events, either of the same type or of different types.")
        classification_rules.append("- Create a separate event object for each event mention.")
        classification_rules.append("- Do not extract events that are explicitly NEGATED (e.g., \"denies pain\", \"couldn\'t sleep\") unless the event is marked as \"Identifed Always\".")
        if attribute_output:
            classification_rules.append("- If a NEGATED event is marked as Identified Always, extract it and set the negation attribute to true.")
        classification_rules.append("- Ignore events that refer to FUTURE or HYPOTHETICAL scenarios (e.g., \"will eat tomorrow\").")
        classification_rules.append("- A valid event must have occurred in the recent past or be occurring at the time of writing.")
        classification_rules.append("- If no events are found, or if uncertain, return: {\"events\": []}")
        classification_rules = "\n".join(classification_rules)
    elif prompt_version == 2:
        classification_rules.append("If the event talks about a patient's history or future wish of an event, DO NOT EXTRACT that event.")
        classification_rules.append("Consider events only if they relate to the patient themselves (e.g., exclude caregivers or family members' own experiences).")
        classification_rules = " ".join(classification_rules)
    elif prompt_version == 3:
        classification_rules.append("If the event talks about a patient's history (before the shift) or future plan of an event (after the shift), DO NOT EXTRACT that event.")
        classification_rules.append("Consider events ONLY if they relate to the patient (e.g., exclude caregivers' or family members' own experiences of Sleep/Excretion/Pain/Eating/Family).")
        classification_rules = " ".join(classification_rules)
    elif prompt_version == 4:
        classification_rules.append("If the event talks about a patient's history (before the shift) or future plan/request of an event (after the shift), DO NOT EXTRACT that event.")
        classification_rules.append("Consider events ONLY if they relate to the patient as the actor (e.g., exclude caregivers' or family members' own experiences).")
        classification_rules.append("If the text has unclear actor, assume it as patient.")
    elif prompt_version == 5:
        classification_rules.append("If the event talks about a patient's history (before the shift) or future plan/request of an event (after the shift), DO NOT EXTRACT that event.")
        classification_rules.append("Consider events ONLY if they relate to the patient as the actor (e.g., exclude caregivers' or family members' own experiences).")
        classification_rules.append("""An event is extracted from the text even if it is negated (e.g. "did not sleep"), but the respective negation attribute is set to True.""")
        classification_rules.append("If the text has unclear actor, assume it as patient.")
    return classification_rules

def get_output_format(predefined_event_names, attribute_output, prompt_version):
    if prompt_version == 1:
        attribute_clause = (""", "attributes": {
                                        "<attribute_name>": "<attribute_value>"
                                        }""") if attribute_output else ""
        output_format = f"""{{
                                "events": [
                                    {{
                                    "event_id": "< A unique id eg.: e1 | e2 |...>"
                                    "event_type": "<{" | ".join(predefined_event_names)}>",
                                    "text_quote": "<fragement from the text indicating the event_type>"
                                    {attribute_clause}
                                    }}
                                ],
                                "order": [
                                    ["e1", "before" | "after" | "simultaneous", "e2" ]
                                ]
                                }}"""  
    elif prompt_version in [2,3,4,5]:
        attribute_specs = ""
        if attribute_output:
            attribute_specs = """
            "event_attributes": { 
                            // Only extract attributes for events present in the text:
                            "Sleep": { 
                            "quality": string (e.g., poor, good, etc.),
                            "duration": string (e.g., short, on and off, etc.),
                            }, 
                            "Excretion": { 
                            "type": string (e.g., urine, stool, etc.),
                            "frequency": string (e.g., 2x, rare, etc.),
                            "quality": string (e.g., hard, yellow, etc.)
                            }, 
                            "Family": { 
                            "interaction": string (e.g., visit, call, communication, etc.), 
                            "relation": string (e.g., mother, son, etc.)
                            }, 
                            "Pain": { 
                            "severity": string (e.g., mild, moderate, severe, numeric scale if present, etc.), 
                            "location": string (e.g., right knee, head, etc.), 
                            "duration": string (e.g., all night, 3 hours, etc.)
                            }, 
                            "Eating": { 
                            "food": string, (e.g., pancakes, porridge, etc.)
                            "amount": string, (e.g., 1, 2 bowls, etc.)
                            "method": string (e.g., oral, tube, etc.)
                            }"""
        if prompt_version in [2,3]:
            output_format = """
                        {   
                        "case_attributes":[ // all properties of the patients case that ocurred before the shift or at home or patient history.
                            {
                            "attribute_name":"attribute_value" // attributes of the case such as patient history.
                            }
                        ]  
                        "events": [ //events occurreing during the shift of the clinical narrative
                            { 
                            "event_id": string, ("e1", "e2", etc.)
                            "event_type": string ("Sleep", "Excretion", "Eating", "Family", "Pain", "Unknown"), 
                            "text_quote": string (fragment of the text from which attributes are extracted), 
                            "actor": string ("patient", "family member", "others"),
                            "negation": boolean, (true if the event is negated e.g., did not sleep. false otherwise)
                            "time": string (e.g., am, morning, 5pm . default: "Unknown"), 
                            "caused_by": string (name of another event that caused this event. default: "Unknown"), 
                            """ + attribute_specs + """ 
                            } 
                            } 
                        ], 
                        "order": [ //partial order of extracted events
                            { 
                            "event_id_1": string, (e1, e2...)
                            "relation": string ("before", "after", "simultaneous", "unknown"), 
                            "event_id_2": string (e1, e2...)
                            } 
                        ],
                        
                        }"""
        elif prompt_version in [4,5]:
            output_format = """
                        {   
                        "case_attributes":[ // all properties of the patients case that ocurred before the shift or at home or patient history or plan.
                            {
                            "attribute_name":"attribute_value" // attributes of the case such as patient history.
                            }
                        ]  
                        "events": [ //events occurred/occurring DURING THE SHIFT
                            { 
                            "event_id": string, ("e1", "e2", etc.)
                            "event_type": string ("Sleep", "Excretion", "Eating", "Family", "Pain", "Unknown"), 
                            "text_quote": string (fragment of the text from which attributes are extracted), 
                            "actor": string ("patient", "family member", "others"),
                            "object": string ("patient", "family member", "others"),
                            "negation": boolean, (true if the event is negated e.g., did not sleep. false otherwise)
                            "time": string (e.g., am, morning, 5pm . default: "Unknown"), 
                            "caused_by": string (name of another event that caused this event. default: "Unknown"), 
                            """ + attribute_specs + """ 
                            } 
                            } 
                        ], 
                        "order": [ //partial order of extracted events
                            { 
                            "event_id_1": string, (e1, e2...)
                            "relation": string ("before", "after", "simultaneous", "unknown"), 
                            "event_id_2": string (e1, e2...)
                            } 
                        ],
                        
                        }"""
    return output_format

def get_output_rules(attribute_output, prompt_version):
    if prompt_version == 1:
        output_rules =  f"""
                    Rules:
                        - Ensure the output is valid JSON (parseable).
                        - "eventS" must always be an array (can be empty).
                        - Multiple instances of the same event type appear with different ids.
                        - Events appear in the array in the same order in which they appear in the text.
                        - Their partial orders can be expressed using the "order" section.
                        - Do not include extra keys, comments, or text.
                        {'- Each object in "events" must contain "event_type", "text_quote", and "attributes".' if attribute_output else '- Each object in "events" must contain "event_type" and "text_quote".' }
                        {'- If an event attribute type has no value mentioned, return "attributes": {"< attribute name >:Unknown"} for each attribute type defined for the event type' if attribute_output else ''}
                        """
    elif prompt_version in [2,3]:
        output_rules = f"""
                        **Negation Policy:** 
                        * For events like Sleep, Pain, Excretion, and Eating, the event is recorded in the output even if it is negated 
                        (e.g. "did not sleep"). 
                        * For events like Family, if the event is negated (e.g. "no family member visited"), it is ignored and not 
                        included in the output. 
                        """
    elif prompt_version in [4]:
        output_rules = ''
    elif prompt_version in [5]:
        output_rules = """**Note:**
                        - Sedation and resting are not a sleep event.
                        - Sleep apnea occurs during sleep.
                        - people 'wake' up from sleep.
                        """
    return output_rules