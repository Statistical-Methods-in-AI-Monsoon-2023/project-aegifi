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

col1,mid_col, col2 = st.columns((1,1,1))

with col1:
    models = st.multiselect(
        'Select the models you want to run inference on:',
        ['Binary Naive Bayes', 'Multinomial Naive Bayes', 'XGBoost', 'Binary GRU', 'Rank GRU', 'Multinomial GRU'],
    )

with mid_col:
    xgb_word_embeddings = []
    bnb_word_embeddings = []
    mnb_word_embeddings = []
    if 'XGBoost' in models:
        # add a checkboxes for word embeddings
        with st.expander('Select the word embeddings you want to use with XGBoost:'):
            xgb_w2v = st.checkbox('Word 2 Vec', value=True, key='xgb_w2v')
            xgb_bow = st.checkbox('Bag of Words', key='xgb_bow')
            xgb_tfidf = st.checkbox('TF-IDF', key='xgb_tfidf')
            xgb_tf_w2v = st.checkbox('TF-IDF Weighted Word 2 Vec', key='xgb_tf_w2v')
            
            if xgb_w2v:
                xgb_word_embeddings.append('Word 2 Vec')
            if xgb_bow:
                xgb_word_embeddings.append('Bag of Words')
            if xgb_tfidf:
                xgb_word_embeddings.append('TF-IDF')
            if xgb_tf_w2v:
                xgb_word_embeddings.append('TF-IDF Weighted Word 2 Vec')
    
    if 'Binary Naive Bayes' in models:
        with st.expander('Select the word embeddings you want to use with Binary Naive Bayes:'):
            # add a checkboxes for word embeddings
            bnb_w2v = st.checkbox('Word 2 Vec', value=True, key='bnb_w2v')
            bnb_bow = st.checkbox('Bag of Words', key='bnb_bow')
            bnb_tfidf = st.checkbox('TF-IDF', key='bnb_tfidf')
            bnb_tf_w2v = st.checkbox('TF-IDF Weighted Word 2 Vec', key='bnb_tf_w2v')
            
            if bnb_w2v:
                bnb_word_embeddings.append('Word 2 Vec')
            if bnb_bow:
                bnb_word_embeddings.append('Bag of Words')
            if bnb_tfidf:
                bnb_word_embeddings.append('TF-IDF')
            if bnb_tf_w2v:
                bnb_word_embeddings.append('TF-IDF Weighted Word 2 Vec')

    if 'Multinomial Naive Bayes' in models:
        with st.expander('Select the word embeddings you want to use with Multinomial Naive Bayes:'):
            # add a checkboxes for word embeddings
            mnb_bow = st.checkbox('Bag of Words', key='mnb_bow', value=True)
            mnb_tfidf = st.checkbox('TF-IDF', key='mnb_tfidf')
            
            if mnb_bow:
                mnb_word_embeddings.append('Bag of Words')
            if mnb_tfidf:
                mnb_word_embeddings.append('TF-IDF')

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
    'Bag of Words': 'bow',
    'TF-IDF Weighted Word 2 Vec': 'tf_w2v'
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
            xgb_word_embeds = [embed_code[word_embeddings] for word_embeddings in xgb_word_embeddings]
            bnb_word_embeds = [embed_code[word_embeddings] for word_embeddings in bnb_word_embeddings]
            mnb_word_embeds = [embed_code[word_embeddings] for word_embeddings in mnb_word_embeddings]
            for model in models:
                if model == 'XGBoost':
                    for word_embeddings in xgb_word_embeds:
                        streamlit_run(model_code[model], word_embeddings=word_embeddings, load_models=True)
                elif model == 'Binary Naive Bayes':
                    for word_embeddings in bnb_word_embeds:
                        streamlit_run(model_code[model], word_embeddings=word_embeddings, load_models=True)
                elif model == 'Multinomial Naive Bayes':
                    for word_embeddings in mnb_word_embeds:
                        streamlit_run(model_code[model], word_embeddings=word_embeddings, load_models=True)
                # else:
                #     streamlit_run(model_code[model], word_embeddings='w2v', load_models=True)
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

        st.markdown('## Metrics')
        st.markdown('---')

        # divide metrics into 2 different dicts
        dfs_1 = {}
        dfs_2 = {}
        count = 0
        for metric in dfs:
            if count < 4:
                dfs_1[metric] = {
                    'df': dfs[metric],
                    'col_idx': count % 4
                }
            else:
                dfs_2[metric] = {
                    'df': dfs[metric],
                    'col_idx': count % 4
                }
            count += 1
        
        # display the metrics in 4 columns
        cols_1 = st.columns(4)
        for metric in dfs_1:
            with cols_1[dfs_1[metric]['col_idx']]:
                st.markdown(f'### {metric}')
                st.bar_chart(dfs_1[metric]['df'], x='Model', y='Value')
        
        cols_2 = st.columns(4)
        for metric in dfs_2:
            with cols_2[dfs_2[metric]['col_idx']]:
                st.markdown(f'### {metric}')
                st.bar_chart(dfs_2[metric]['df'], x='Model', y='Value')
            

if run_training:
    with col2:
        with st.spinner('Training in progress...'):
            pass
            # streamlit_run([model_code[model] for model in models], word_embeddings=embed_code[word_embeddings], load_models=False)
        st.balloons()