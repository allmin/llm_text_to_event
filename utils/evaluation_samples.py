from glob import glob
import os,sys
import pandas as pd
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

focus_ids = {}
for focus in ["Doc", "Sent"]:
    if focus == "Doc":
        uid = "ROW_ID"
    elif focus == "Sent":
        uid = "UID"
    files = glob(f"../exports/06_evaluation_samples/*_Event_Name_LLM_Events_example_evidence_{focus}_Sleep.xlsx")
    full_df = []
    for file in files:
        file_name = os.path.basename(file)
        df = pd.read_excel(file)
        if file_name[:2] in ["FP","FN"]:
            sample_per = 0.9
        else:
            sample_per = 0.1
        N = len(df)
        subN = int(sample_per*N)
        finalN= max(1,subN)
        print(file_name, finalN)
        df_samples = df.sample(n=finalN,replace=False,random_state=42)
        full_df.append(df_samples)
    full_df = pd.concat(full_df).reset_index()
    unique_row_ids = full_df[uid].astype(str).tolist()
    focus_ids[focus] = unique_row_ids

        