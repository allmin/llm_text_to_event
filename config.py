llm_type = "llama3.1:70b"
event_types = ["Pain", "Sleep", "Excretion", "Eating", "Family"]
event_description_dict_embedder = {"Eating":"To take food into the body by mouth",
                              "Excretion":"Waste matter discharged from the body s feces or urine",
                              "Family":"A visit, call or communication with a member of the family", #Interaction with a family member
                              "Pain":"The reporting of pain or an observation of pain signals by the doctor/nurse",
                              "Sleep":"The act of sleeping, possibly mentioning its quality or quantity"}

event_description_dict_llm = {
                            "Eating": "The patient takes food into their body by mouth.",
                            "Excretion": "The patient discharges waste matter from their body.",
                            "Family": "The patient has a visit, call, or communication with a family member.",
                            "Pain": "The patient reports or shows signs of pain.",
                            "Sleep": "The patient is sleeping, or the sleep’s quality or quantity is described."
                            }

event_attributes_dict_llm = {
                                "Eating": {
                                    "food": "<string>",
                                    "amount": "<string>",
                                    "method": "<string>  # e.g., oral, tube"
                                },
                                "Excretion": {
                                    "type": "<string> # e.g., urine, stool",
                                    "frequency": "<string>",
                                    "quality": "<string> # e.g., loose, hard"
                                },
                                "Family": {
                                    "interaction": "<string> # e.g., visit, call, communication",  
                                    "relation": "<string> # e.g., mother, son"
                                },
                                "Pain": {
                                    "severity": "<string> # e.g., mild, moderate, severe, numeric scale if present",
                                    "location": "<string>",
                                    "duration": "<string>"
                                },
                                "Sleep": {
                                    "quality": "<string> # e.g., good, poor, restless",   
                                    "duration": "<string> # e.g., hours, phrases like \"all night\""   
                                }
                            }

examples = """
                ---
                Examples:
                
                sentence: "Patient ate breakfast this morning."
                output: {
                "events": [
                    {
                    "event_type": "Eating",
                    "attributes": {"food": "breakfast"}
                    }
                ]
                }
                
                sentence: "Patient reported severe abdominal pain."
                output: {
                "events": [
                    {
                    "event_type": "Pain",
                    "attributes": {"severity": "severe", "location": "abdominal"}
                    }
                ]
                }
                
                sentence: "Patient called his son."
                output: {
                "events": [
                    {
                    "event_type": "Family",
                    "attributes": {"interaction": "call", "relation": "son"}
                    }
                ]
                }
                
                sentence: "Patient had a loose stool overnight."
                output: {
                "events": [
                    {
                    "event_type": "Excretion",
                    "attributes": {"type": "stool", "quality": "loose"}
                    }
                ]
                }
                
                sentence: "Patient was able to sleep well last night."
                output: {
                "events": [
                    {
                    "event_type": "Sleep",
                    "attributes": {"quality": "well"}
                    }
                ]
                }
                
                sentence: "The patient couldn\'t sleep due to severe pain."
                output: {
                "events": [
                    {
                    "event_type": "Sleep",
                    "attributes": {"quality": "poor"}
                    },
                    {
                    "event_type": "Pain",
                    "attributes": {"severity": "severe"}
                    }
                ]
                }
                
                sentence: "The patient complained of severe back pain, was given Tylenol, but the pain persisted and he was then prescribed stronger morphine."
                output: {
                "events": [
                    {
                    "event_type": "Pain",
                    "attributes": {"severity": "severe", "location": "back"}
                    },
                    {
                    "event_type": "Pain",
                    "attributes": {"duration": "persistent"}
                    }
                ]
                }
                """