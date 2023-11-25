import streamlit as st
import sys
import os
from PIL import Image

sys.path.append('./src/')
sys.path.append('./utils/')

from main import streamlit_run
from utils import MetricReader

adyansh = Image.open('./assets/adyansh.png')
akshit = Image.open('./assets/akshit.png')

st.set_page_config(layout="wide")

header_col_1,_, header_col_2, header_col_3 = st.columns((2,1,1,1))


with header_col_1:
    st.title("Team 13 - AegiFi")

    # header
    st.markdown('## Predicting Movie Genres from Plot Summaries')

with header_col_2:
    st.image(adyansh, caption='Adyansh Kakran', width=200)
with header_col_3:
    st.image(akshit, caption='Akshit Sinha', width=200)
    
st.markdown('---')

col1,mid_col, col2 = st.columns((1,1,1))

with col1:
    st.markdown('### Model Selection')
    models = st.multiselect(
        'Select the models you want to run inference on:',
        ['Binary Naive Bayes', 'Multinomial Naive Bayes', 'XGBoost', 'Binary GRU', 'Rank GRU', 'Multinomial GRU'],
    )

with mid_col:
    st.markdown('### Word Embeddings Selection')
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
            xgb_d2v = st.checkbox('Doc 2 Vec', key='xgb_d2v')
            
            if xgb_w2v:
                xgb_word_embeddings.append('Word 2 Vec')
            if not xgb_w2v and 'Word 2 Vec' in xgb_word_embeddings:
                xgb_word_embeddings.remove('Word 2 Vec')
                
            if xgb_bow:
                xgb_word_embeddings.append('Bag of Words')
            if not xgb_bow and 'Bag of Words' in xgb_word_embeddings:
                xgb_word_embeddings.remove('Bag of Words')

            if xgb_tfidf:
                xgb_word_embeddings.append('TF-IDF')
            if not xgb_tfidf and 'TF-IDF' in xgb_word_embeddings:
                xgb_word_embeddings.remove('TF-IDF')    
            
            if xgb_tf_w2v:
                xgb_word_embeddings.append('TF-IDF Weighted Word 2 Vec')
            if not xgb_tf_w2v and 'TF-IDF Weighted Word 2 Vec' in xgb_word_embeddings:
                xgb_word_embeddings.remove('TF-IDF Weighted Word 2 Vec')
                
            if xgb_d2v:
                xgb_word_embeddings.append('Doc 2 Vec')
            if not xgb_d2v and 'Doc 2 Vec' in xgb_word_embeddings:
                xgb_word_embeddings.remove('Doc 2 Vec')
    
    if 'Binary Naive Bayes' in models:
        with st.expander('Select the word embeddings you want to use with Binary Naive Bayes:'):
            # add a checkboxes for word embeddings
            bnb_w2v = st.checkbox('Word 2 Vec', value=True, key='bnb_w2v')
            bnb_bow = st.checkbox('Bag of Words', key='bnb_bow')
            bnb_tfidf = st.checkbox('TF-IDF', key='bnb_tfidf')
            bnb_tf_w2v = st.checkbox('TF-IDF Weighted Word 2 Vec', key='bnb_tf_w2v')
            bnb_d2v = st.checkbox('Doc 2 Vec', key='bnb_d2v')
            
            if bnb_w2v:
                bnb_word_embeddings.append('Word 2 Vec')
            if not bnb_w2v and 'Word 2 Vec' in bnb_word_embeddings:
                bnb_word_embeddings.remove('Word 2 Vec')
                
            if bnb_bow:
                bnb_word_embeddings.append('Bag of Words')
            if not bnb_bow and 'Bag of Words' in bnb_word_embeddings:
                bnb_word_embeddings.remove('Bag of Words')

            if bnb_tfidf:
                bnb_word_embeddings.append('TF-IDF')
            if not bnb_tfidf and 'TF-IDF' in bnb_word_embeddings:
                bnb_word_embeddings.remove('TF-IDF')    
            
            if bnb_tf_w2v:
                bnb_word_embeddings.append('TF-IDF Weighted Word 2 Vec')
            if not bnb_tf_w2v and 'TF-IDF Weighted Word 2 Vec' in bnb_word_embeddings:
                bnb_word_embeddings.remove('TF-IDF Weighted Word 2 Vec')
                
            if bnb_d2v:
                bnb_word_embeddings.append('Doc 2 Vec')
            if not bnb_d2v and 'Doc 2 Vec' in bnb_word_embeddings:
                bnb_word_embeddings.remove('Doc 2 Vec')

    if 'Multinomial Naive Bayes' in models:
        with st.expander('Select the word embeddings you want to use with Multinomial Naive Bayes:'):
            # add a checkboxes for word embeddings
            mnb_bow = st.checkbox('Bag of Words', key='mnb_bow', value=True)
            mnb_tfidf = st.checkbox('TF-IDF', key='mnb_tfidf')

            if mnb_bow:
                mnb_word_embeddings.append('Bag of Words')
            if not mnb_bow and 'Bag of Words' in mnb_word_embeddings:
                mnb_word_embeddings.remove('Bag of Words')

            if mnb_tfidf:
                mnb_word_embeddings.append('TF-IDF')
            if not mnb_tfidf and 'TF-IDF' in mnb_word_embeddings:
                mnb_word_embeddings.remove('TF-IDF')    

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
    'TF-IDF Weighted Word 2 Vec': 'tf_w2v',
    'Doc 2 Vec': 'd2v'
}

rev_embed_code = {
    'w2v': 'Word 2 Vec',
    'tfidf': 'TF-IDF',
    'bow': 'Bag of Words',
    'tf_w2v': 'TF-IDF Weighted Word 2 Vec',
    'd2v': 'Doc 2 Vec'
}


with col1:
    
    st.markdown('#### Run on IMDB Dataset')
    
    inner_col1, inner_col2 = st.columns((1,1))
    with inner_col1:
        run_training = st.button('Run Training', disabled = len(models) == 0)
    with inner_col2:
        run_inference_test = st.button('Run Inference', disabled = len(models) == 0)

with col2:
    st.markdown('### Create your own summary!')
    # text box to enter sample plot
    sample_plot = st.text_area('Enter a sample plot to run inference on:', height=50)
    run_inference = st.button('Run Inference on Sample Plot', key='sample_plot', disabled = len(sample_plot) == 0)

# enable the button only if at least one model is selected
if run_inference or run_inference_test:
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
            
            model_to_embed = {
                'XGBoost': xgb_word_embeds,
                'Binary Naive Bayes': bnb_word_embeds,
                'Multinomial Naive Bayes': mnb_word_embeds
            }
            
            outputs = []
            if len(sample_plot) == 0:
                sample_plot = None
            
            for model in models:
                if model in ['Binary GRU', 'Rank GRU', 'Multinomial GRU']:
                    out = streamlit_run(model_code[model], word_embeddings=None, load_models=True, plot_sample=sample_plot)
                    
                    outputs.append({
                        'model': model,
                        'word_embeddings': None,
                        'output': out
                    })
                    continue
                    
                for word_embeddings in model_to_embed[model]:
                    out = streamlit_run(model_code[model], word_embeddings=word_embeddings, load_models=True, plot_sample=sample_plot)
                    
                    outputs.append({
                        'model': model,
                        'word_embeddings': word_embeddings,
                        'output': out
                    })
                    
        st.balloons()
    
    if run_inference:
        with col2:
            if len(outputs) > 0:
                st.markdown('### Inference Results')
                st.markdown('---')
                
                for output in outputs:
                    header = ''
                    if output['word_embeddings'] is None:
                        header = f'##### {output["model"]}'
                    else:
                        header = f'{output["model"]} with {rev_embed_code[output["word_embeddings"]]}'
                    with st.expander(header, expanded=True):
                        st.markdown(f'##### Predicted Genres:')
                        for genre in output['output']:
                            st.markdown(f'- {genre}')
                        
    
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