import streamlit as st
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
import pandas as pd

sys.path.append('./src/')
sys.path.append('./utils/')

from main import streamlit_run, lime_run
from utils import MetricReader

if 'outputs' not in st.session_state:
    st.session_state.outputs = []

if 'sample_plot' not in st.session_state:
    st.session_state.sample_plot = ""

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

col1, col2 = st.columns((1,2))

with col1:
    st.markdown('### Model Selection')
    models = st.multiselect(
        'Select the models you want to run inference on:',
        ['Binary Naive Bayes', 'Multinomial Naive Bayes', 'XGBoost', 'Binary GRU', 'Rank GRU', 'Multinomial GRU', 'Transformer'],
    )

with col1:
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
    "Multinomial GRU": "mgru",
    "Transformer": "trf"
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
    
    st.markdown('#### Run on IMDB Dataset for benchmarking results')
    
    inner_col1, inner_col2 = st.columns((1,1))
    with inner_col1:
        run_training = st.button('Run Training', disabled = len(models) == 0)
    with inner_col2:
        run_inference_test = st.button('Run Inference', disabled = len(models) == 0)

with col2:
    st.markdown('### Create your own summary!')
    # text box to enter sample plot
    st.session_state.sample_plot = st.text_area('Enter a sample plot to run inference on:', height=150)
    run_inference = st.button('Run Inference on Sample Plot', key='st.session_state.sample_plot', disabled = len(st.session_state.sample_plot) == 0)

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
            
            if len(st.session_state.sample_plot) == 0:
                st.session_state.sample_plot = None
            else:
                # reset the outputs
                st.session_state.outputs = []
            
            for model in models:
                if model in ['Binary GRU', 'Rank GRU', 'Multinomial GRU', 'Transformer']:
                    out = streamlit_run(model_code[model], word_embeddings=None, load_models=True, plot_sample=st.session_state.sample_plot)
                    
                    if st.session_state.sample_plot:
                        st.session_state.outputs.append({
                            'model': model,
                            'word_embeddings': None,
                            'output': out
                        })
                    continue
                    
                for word_embeddings in model_to_embed[model]:
                    out = streamlit_run(model_code[model], word_embeddings=word_embeddings, load_models=True, plot_sample=st.session_state.sample_plot)
                    
                    if st.session_state.sample_plot:
                        st.session_state.outputs.append({
                            'model': model,
                            'word_embeddings': word_embeddings,
                            'output': out
                        })
                    
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

def map_to_colormap_hex(word_tuples, colormap='RdBu'):
    # get the max and min values
    max_value = max(word_tuples, key=lambda x: x[1])[1]
    min_value = min(word_tuples, key=lambda x: x[1])[1]
    
    # save word-value pairs in a dict
    word_dict = {}
    for word, value in word_tuples:
        word_dict[word] = value
    
    # do min-max normalization
    word_tuples = [(word, (value - min_value) / (max_value - min_value)) for word, value in word_tuples]

    # map the values to 0 - 1 range
    # word_tuples = [(word, (value + 1) / 2) for word, value in word_tuples]
    
    # map to the colormap
    word_tuples = [(word, plt.get_cmap(colormap)(value)) for word, value in word_tuples]
    
    # convert to hex
    hex_values = {}
    for word, value in word_tuples:
        hex_values[word] = {
            'importance': word_dict[word],
            'color': matplotlib.colors.rgb2hex(value)
        }
    
    return hex_values

            
def run_lime(model, word_embeddings, genre):
    with col2:
        if word_embeddings == None or word_embeddings == '':
            st.write(f'### LIME Explanation for {genre} prediction by {model}')
        else:
            st.write(f'### LIME Explanation for {genre} prediction by {model} with {rev_embed_code[word_embeddings]}')
        st.write('---')
        
        with st.spinner('LIME explanation in progress...'):
            out = lime_run(model_code[model], word_embeddings=word_embeddings, genre=genre, plot_sample=st.session_state.sample_plot)
        
        # out is list of tuples of (word, importance) pairs
        print(out)
        # color the words based on their importance and display the entire sample plot with the words colored
        word_color_dict = map_to_colormap_hex(out)
        
        # color the words in the sample plot
        display_sample_plot = st.session_state.sample_plot.replace('\n', ' ')
        # add spaces around punctuation
        display_sample_plot = display_sample_plot.replace('.', ' . ')
        display_sample_plot = display_sample_plot.replace(',', ' , ')
        display_sample_plot = display_sample_plot.replace('?', ' ? ')
        display_sample_plot = display_sample_plot.replace('!', ' ! ')
        display_sample_plot = display_sample_plot.replace('"', ' " ')
        display_sample_plot = display_sample_plot.replace("'", " ' ")
        
        print(word_color_dict)
        
        display_words = display_sample_plot.split(' ')
        for i in range(len(display_words)):
            if display_words[i] in word_color_dict:
                display_words[i] = f'<span style="color: {word_color_dict[display_words[i]]["color"]}">{display_words[i]}</span>'
        
        display_sample_plot = ' '.join(display_words)
        # remove spaces around punctuation
        display_sample_plot = display_sample_plot.replace(' . ', '. ')
        display_sample_plot = display_sample_plot.replace(' , ', ', ')
        display_sample_plot = display_sample_plot.replace(' ? ', '? ')
        display_sample_plot = display_sample_plot.replace(' ! ', '! ')
        display_sample_plot = display_sample_plot.replace(' " ', '" ')
        display_sample_plot = display_sample_plot.replace(" ' ", "' ")
        
        col2_inners = st.columns((3,5))
        
        
        with col2_inners[0]:
            st.markdown(display_sample_plot, unsafe_allow_html=True)
        
        # word_color_dict is a dict of dicts with keys as words and values as dicts with keys as importance and color
        # convert it to a dataframe with the columns as word, importance and color
        df = pd.DataFrame.from_dict(word_color_dict, orient='index')
        df = df.reset_index()
        df = df.rename(columns={'index': 'word'})
        df = df[['word', 'importance', 'color']]
        
        # convert word_color_dict to a dict with word as key and color as value
        word_color_dict = {word: word_color_dict[word]['color'] for word in word_color_dict}
        
        fig = px.bar(df, x='importance', y='word', color='word',
             labels={'importance': 'Importance', 'word': 'Word'},
             title='Importance',
             color_discrete_map=word_color_dict,
             orientation='h')  # 'h' for horizontal bars
        
        with col2_inners[1]:
            st.plotly_chart(fig)
        st.write('---')
        

if st.session_state.outputs is not None:
    with col2:
        if len(st.session_state.outputs) > 0:
            st.markdown('### Inference Results')
            st.markdown('---')
            st.markdown('Click on the genres to see the LIME explanations for the predictions made by the models for that genre')
            
            # divide outputs into lists of 3
            outputs_1 = []
            outputs_2 = []
            outputs_3 = []
            
            # assign each output to a list modulo 3
            count = 0
            for output in st.session_state.outputs:
                if count % 3 == 0:
                    outputs_1.append(output)
                elif count % 3 == 1:
                    outputs_2.append(output)
                else:
                    outputs_3.append(output)
                count += 1
            
            # display the outputs in 3 columns
            cols_1 = st.columns(3)
            for i, output_list in enumerate([outputs_1, outputs_2, outputs_3]):
                for output in output_list:
                    header = ''
                    if output['word_embeddings'] is None or output['word_embeddings'] == '':
                        output['word_embeddings'] = ''
                        header = f'#### {output["model"]}'
                    else:
                        header = f'#### {output["model"]} with {rev_embed_code[output["word_embeddings"]]}'
                    with cols_1[i]:
                        with st.expander(header,expanded=True):
                            # create a button with the output as the label
                            for output_label in output['output']:
                                bt = st.button(output_label, key=f"{output_label}_{output['model']}_{output['word_embeddings']}", on_click=run_lime, args=[output['model'], output['word_embeddings'], output_label]) 

if run_training:
    with col2:
        with st.spinner('Training in progress...'):
            pass
            # streamlit_run([model_code[model] for model in models], word_embeddings=embed_code[word_embeddings], load_models=False)
        st.balloons()