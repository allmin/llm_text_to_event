# streamlit run annotate.py --server.port 8501
# fuser -k 8501/tcp || true; streamlit run annotate.py --server.port 8501
#srun --pty -p gpu_mig -t 01-00:00:00 -N 1 -n 1 --mem=8G streamlit run annotate.py --server.port 8501 --server.address 0.0.0.0
# srun --pty -p gpu_a100 -t 01-00:00:00 -N 1 -n 1 --gpus-per-node 1 --mem=8G bash
# ssh -L 8501:localhost:8501 gcn17

import streamlit as st
import pandas as pd
import re
import os
from glob import glob
all_paths = glob("../exports/04_groundtruth/**/**")
only_folders = [p for p in all_paths if os.path.isdir(p)]
only_folders = [p for p in only_folders if "Annotating" in p]

# Streamlit selectbox
annotation_folder = st.selectbox("Select Annotation Folder", only_folders)
pickle_files = glob(f"{annotation_folder}/*.pkl")
FILE_PATH = st.selectbox("Select Pickle File", pickle_files)

if ("FILE_PATH" not in st.session_state) or st.session_state['FILE_PATH'] != FILE_PATH:
    st.session_state["FILE_PATH"] = FILE_PATH
    st.session_state.ET = os.path.splitext(os.path.basename(FILE_PATH))[0].split("_")[0]
    st.session_state.focus = "Doc" if "Document" in FILE_PATH else "Sent"
    st.write(st.session_state.ET)
 
ET = st.session_state.ET   
focus = st.session_state.focus  # DOCUMENT or Sentence

@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_pickle(path)
    if f"{focus}_gt_{ET}" not in df.columns:
        df[f"{focus}_gt_{ET}"] = None
    if "negation" not in df.columns:
        df["negation"] = False
    if "good_example" not in df.columns:
        df["good_example"] = False
    if "comment" not in df.columns:
        df['comment'] = ''
    df.index = range(len(df))
    return df


def save_data(df):
    global FILE_PATH
    df.to_pickle(FILE_PATH)
    
def get_last_continuous_sequence(numbers):
    if not numbers:
        return []

    sequences = []
    current_seq = [numbers[0]]

    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i - 1] + 1:
            current_seq.append(numbers[i])
        else:
            sequences.append(current_seq)
            current_seq = [numbers[i]]

    sequences.append(current_seq)  # append the last sequence

    return sequences

st.title(f"🛌 {focus}: {ET} Ground Truth Correction Tool")

df = load_data(st.session_state.FILE_PATH)

page_size = 10
total_pages = len(df) // page_size + int(len(df) % page_size > 0)

# Track current page in session state
if "page_number" not in st.session_state:
    st.session_state.page_number = 1

page = st.session_state.page_number
start_idx = (page - 1) * page_size
end_idx = min(start_idx + page_size, len(df))
subset = df.iloc[start_idx:end_idx].copy()


# a dropdown to select page
page_selection = st.selectbox("Select Page", range(1, total_pages + 1), index=int(page) - 1)
if page_selection != page:
    st.session_state.page_number = page_selection
    st.rerun()


#find rows where gt_ET is None and find the first row index
if st.button("Show unlabelled row ids"):
    none_rows = df[df[f"{focus}_gt_{ET}"].isnull()]
    none_rows_index = none_rows.index.tolist()
    none_rows_index = [i for i in none_rows_index if i > (page*10)]
    none_rows = df.loc[none_rows_index]
    
    
    if not none_rows.empty:
        first_none_index = none_rows.index[0]
        if first_none_index < start_idx or first_none_index >= end_idx:
            st.session_state.page_number = first_none_index // page_size + 1
            st.rerun()

st.write(f"Showing rows {start_idx} to {end_idx - 1} of {len(df)}")

# Auto-saving corrections
for i, row in subset.iterrows():
    st.markdown(f"**Row {i}** — UID: `{row['UID']}` — Keyword: `{row['Keyword']}`")

    # Highlight keyword in sentence
    pattern = re.escape(str(row["Keyword"]))
    if focus == "Doc":
        text_to_highlight = str(row["Document"])
    else:
        text_to_highlight = str(row["Sentence"])
    highlighted = re.sub(pattern, f"**:orange[{row['Keyword']}]**", text_to_highlight, flags=re.IGNORECASE)
    st.markdown(highlighted, unsafe_allow_html=True)

    current_val = row[f"{focus}_gt_{ET}"]
    options = {
        "Yes": True,
        "No": False,
        "Not Sure": None
    }

    # Determine index of current_val for preselection
    reverse_options = {v: k for k, v in options.items()}
    selected_label = reverse_options.get(current_val, "Not Sure")

    new_label = st.radio(
        f"Is {ET}-related? (row {i}):",

        options=list(options.keys()),
        index=list(options.keys()).index(selected_label),
        key=f"radio_{page}_{i}",
        horizontal=True
    )

    new_value = options[new_label]

    if new_value != current_val:
        df.at[i, f"{focus}_gt_{ET}"] = new_value
        if focus == "Doc":
            df.loc[df['Document'] == row['Document'], f"{focus}_gt_{ET}"] = new_value
        else:
            df.loc[df['Sentence'] == row['Sentence'], f"{focus}_gt_{ET}"] = new_value
        save_data(df)
        st.cache_data.clear()
        st.rerun()
    
    
    current_good_example = df.loc[i, "good_example"]
    current_negation = df.loc[i, "negation"]
    current_comment = df.loc[i, "comment"]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        new_negation = st.checkbox("is event negated?", value=current_negation, key=f"negation_{page}_{i}")
    with col2:
        new_good_example = st.checkbox("Good Example For Paper", value=current_good_example, key=f"good_example_{page}_{i}")
    with col3:
        new_comment = st.text_input("Comment", value=current_comment, key=f"comment_{page}_{i}")
    
    
    if new_good_example != current_good_example:
        df.at[i, "good_example"] = new_good_example
        save_data(df)
        st.cache_data.clear()
    
    if new_negation != current_negation:
        df.at[i, "negation"] = new_negation
        save_data(df)
        st.cache_data.clear()
    
    if new_comment != current_comment:
        df.at[i, "comment"] = new_comment
        save_data(df)
        st.cache_data.clear()

    st.markdown("---")

# 🔽 Pagination Buttons
st.markdown("### Navigation")
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("⬅️ First Page") and st.session_state.page_number > 1:
        st.session_state.page_number = 1
        st.rerun()
with col2:
    if st.button("⬅️ Previous") and st.session_state.page_number > 1:
        st.session_state.page_number -= 1
        st.rerun()
with col3:
    if st.button("Next ➡️") and st.session_state.page_number < total_pages:
        st.session_state.page_number += 1
        st.rerun()
