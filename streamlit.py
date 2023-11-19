import streamlit as st
import sys
import os

sys.path.append('./src/')
sys.path.append('./utils/')

from main import streamlit_run
from utils import MetricReader

st.set_page_config(layout="wide")

st.title("Team 13 - AegiFi")

# header
st.markdown('## Predicting Movie Genres from Plot Summaries')

st.markdown('---')

col1,_, col2 = st.columns((2,1,2))

with col1:
    models = st.multiselect(
        'Select the models you want to run inference on:',
        ['Binary Naive Bayes', 'Multinomial Naive Bayes', 'XGBoost', 'Binary GRU', 'Rank GRU', 'Multinomial GRU'],
    )

    word_embeddings = 'Word 2 Vec'
    if 'XGBoost' in models:
        # add a button to select the word embeddings with word2vec selected by default
        word_embeddings = st.radio(
            'Select the word embeddings to use with XGBoost:',
            ('Word 2 Vec', 'TF-IDF', 'Bag of Words'),
            index=0
        )

model_code = {
    "Binary Naive Bayes": "bnb",
    "Multinomial Naive Bayes": "mnb",
    "XGBoost": "xgb",
    "Binary GRU": "bgru",
    "Rank GRU": "rgru",
    "Multinomial GRU": "mgru"
}

embed_code = {
    'Word 2 Vec': 'w2v',
    "TF-IDF": 'tfidf',
    'Bag of Words': 'bow'
}


with col2:
    inner_col1, inner_col2 = st.columns((1,1))
    with inner_col1:
        run_training = st.button('Run Training', disabled = len(models) == 0)
    with inner_col2:
        run_inference = st.button('Run Inference', disabled = len(models) == 0)

# enable the button only if at least one model is selected

if run_inference:
    # read all the file names in the metrics folder of all subfolders in src folder
    metrics_files = []
    for root, dirs, files in os.walk('./src/'):
        for file in files:
            if file.endswith('.txt'):
                metrics_files.append(os.path.join(root, file))
    
    
    with col2:
        with st.spinner('Inference in progress...'):
            streamlit_run([model_code[model] for model in models], word_embeddings=embed_code[word_embeddings], load_models=True)
        st.balloons()
    
        # check if new metrics files have been added
        new_metrics_files = []
        for root, dirs, files in os.walk('./src/'):
            for file in files:
                if file.endswith('.txt'):
                    if os.path.join(root, file) not in metrics_files:
                        new_metrics_files.append(os.path.join(root, file))

    # open the new metrics files and display the contents
    if len(new_metrics_files) > 0:
        metric_reader = MetricReader(new_metrics_files)
        dfs = metric_reader.read_files()
    
        # there are 8 dfs for 8 metrics
        # create bar charts for each metric
        # 4 charts per row
        num_rows = 2
        num_cols = 4
        rows = []
        for i in range(num_rows):
            rows.append(st.columns(num_cols))
        
        for i in range(num_rows):
            for j in range(num_cols):
                with rows[i][j]:
                    st.bar_chart(dfs[i*num_cols + j], x='Model', y='Value')

if run_training:
    with col2:
        with st.spinner('Training in progress...'):
            streamlit_run([model_code[model] for model in models], word_embeddings=embed_code[word_embeddings], load_models=False)
        st.ballons()