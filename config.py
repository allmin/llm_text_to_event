llm_type = "llama3.1:70b"
event_types = ["Pain", "Sleep", "Excretion", "Eating", "Family"]
# event_description_dict_embedder = {"Eating":"To take food into the body by mouth",
#                               "Excretion":"Waste matter discharged from the body s feces or urine",
#                               "Family":"A visit, call or communication with a member of the family", #Interaction with a family member
#                               "Pain":"The reporting of pain or an observation of pain signals by the doctor/nurse",
#                               "Sleep":"The act of sleeping, possibly mentioning its quality or quantity"}

event_description_dict_llm = {
                            "Eating": "The patient takes food into their body by mouth.",
                            "Excretion": "The patient discharges waste matter from their body.",
                            "Family": "The patient has a visit, call, or communication with a family member.",
                            "Pain": "The patient reports or shows signs of pain.",
                            "Sleep": "The patient is sleeping, or the sleep’s quality or quantity is described."
                            }

event_description_dict_embedder = event_description_dict_llm

common_attributes = {"negation": "<boolean> # true if the event is negated, false otherwise",
                     "time": "<string> # e.g., 2 am, 5 pm, night, after shower, etc.",
                     "caused_by": "<string> # e.g., medication, treatment, etc.",
                     }

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

for et in event_types:
    event_attributes_dict_llm[et].update(common_attributes)



examples = """
                ---
                Examples:
                
                text: "Patient ate breakfast this morning. He seems less anxious."
                output: {
                "events": [
                    {
                    "event_type": "Eating",
                    "text_quote":"Patient ate breakfast this morning"
                    }
                ]
                }
                
                text: "Patient reported severe abdominal pain."
                output: {
                "events": [
                    {
                    "event_type": "Pain",
                    "text_quote": "severe abdominal Pain"
                    }
                ]
                }
                
                text: "Patient called his son around 3 pm."
                output: {
                "events": [
                    {
                    "event_type": "Family",
                    "text_quote": "called his son around 3 pm"
                    }
                ]
                }
                
                text: "Patient had a loose stool overnight."
                output: {
                "events": [
                    {
                    "event_type": "Excretion",
                    "text_quote": "loose stool overnight"
                    }
                ]
                }
                
                text: "Patient was able to sleep well last night."
                output: {
                "events": [
                    {
                    "event_type": "Sleep",
                    "text_quote": "sleep well last night"
                    }
                ]
                }
                
                text: "The patient couldn\'t sleep due to severe pain."
                output: {
                "events": [
                    {
                    "event_type": "Pain",
                    "text_quote": "severe pain"
                    }
                ]
                }
                
                text: "The patient complained of severe back pain, was given Tylenol, but the pain persisted and he was then prescribed stronger morphine."
                output: {
                "events": [
                    {
                    "event_type": "Pain",
                    "text_quote": "complained of severe back pain"
                    },
                    {
                    "event_type": "Pain",
                    "text_quote": "pain persisted"
                    }
                ]
                }
                """

examples_Ao = """
                ---
                Examples:
                
                text: "Patient ate breakfast this morning. He seems less anxious."
                output: {
                "events": [
                    {
                    "event_type": "Eating",
                    "text_quote":"Patient ate breakfast this morning",
                    "attributes": {"food": "breakfast", 
                                   "amount": "Unknown",
                                   "method": "Unknown",
                                   "time":"morning",
                                   "caused_by":"Unknown"}
                    }
                ]
                }
                
                text: "Patient reported severe abdominal pain."
                output: {
                "events": [
                    {
                    "event_type": "Pain",
                    "text_quote": "severe abdominal Pain",
                    "attributes": {"severity": "severe", 
                                   "location": "abdominal",
                                   "duration": "Unknown",
                                   "time":"Unknown",
                                   "caused_by":"Unknown"}
                    }
                ]
                }
                
                text: "Patient called his son around 3 pm."
                output: {
                "events": [
                    {
                    "event_type": "Family",
                    "text_quote": "called his son around 3 pm",
                    "attributes": {"interaction": "call", 
                                    "relation": "son",
                                    "time":"3 pm",
                                    "caused_by":"Unknown"}
                    }
                ]
                }
                
                text: "Patient had a loose stool overnight."
                output: {
                "events": [
                    {
                    "event_type": "Excretion",
                    "text_quote": "loose stool overnight",
                    "attributes": {"type": "stool", 
                                   "quality": "loose", 
                                   "frequency":"overnight", 
                                   "time":"night",
                                   "caused_by":"Unknown"}
                    }
                ]
                }
                
                text: "Patient was able to sleep well last night."
                output: {
                "events": [
                    {
                    "event_type": "Sleep",
                    "text_quote": "sleep well last night",
                    "attributes": {"quality": "well",
                                    "duration": "Unknown", 
                                    "time":"night",
                                    "caused_by":"Unknown"}
                    }
                ]
                }
                
                text: "The patient couldn\'t sleep due to severe pain."
                output: {
                "events": [
                    {
                    "event_type": "Sleep",
                    "text_quote": "couldn\'t sleep",
                    "attributes": {"quality": "poor", 
                                    "duration": "Unknown",
                                   "time":"Unknown",
                                   "caused_by":"Pain"}
                    },
                    {
                    "event_type": "Pain",
                    "text_quote": "severe pain",
                    "attributes": {"severity": "severe", 
                                    "location": "Unknown", 
                                    "duration": "Unknown", 
                                   "time":"before Sleep",
                                   "caused_by":"Unknown"}
                    }
                ]
                }
                
                text: "The patient complained of severe back pain, was given Tylenol, but the pain persisted and he was then prescribed stronger morphine."
                output: {
                "events": [
                    {
                    "event_type": "Pain",
                    "text_quote": "complained of severe back pain",
                    "attributes": {"severity": "severe", 
                                    "location": "back",
                                    "duration": "Unknown", 
                                    "time":"Unknown",
                                    "caused_by":"Unknown"}
                    },
                    {
                    "event_type": "Pain",
                    "text_quote": "pain persisted",
                    "attributes": {"severity": "severe", 
                                   "location": "back",
                                   "duration": "persistent", 
                                   "time":"after Tylenol",
                                   "caused_by":"Unknown"}
                    }
                ]
                }
                """