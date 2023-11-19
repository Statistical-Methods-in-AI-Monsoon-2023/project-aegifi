import streamlit as st
import sys
import os

sys.path.append('./src/')
sys.path.append('./utils/')

from main import streamlit_run
st.title("Team 13 - AegiFi")

st.markdown('---')

models = st.multiselect(
    'Select the models you want to run inference on:',
    ['Binary Naive Bayes', 'Multinomial Naive Bayes', 'XGBoost', 'Binary GRU', 'Rank GRU', 'Multinomial GRU'],
)

# show the selected models as bullet points
st.markdown('#### Selected models:')
st.markdown('\n'.join(['- ' + model for model in models]))

if 'XGBoost' in models:
    # add a button to select the word embeddings with word2vec selected by default
    word_embeddings = st.radio(
        'Select the word embeddings to use:',
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

# add button to run inference 
run_inference = st.button('Run Inference', disabled = len(models) == 0)

# enable the button only if at least one model is selected

if run_inference:
    # read all the file names in the metrics folder of all subfolders in src folder
    metrics_files = []
    for root, dirs, files in os.walk('./src/'):
        for file in files:
            if file.endswith('.txt'):
                metrics_files.append(os.path.join(root, file))
    
    print(metrics_files)
    
    st.write('Runnning inference...')
    streamlit_run([model_code[model] for model in models], word_embeddings=embed_code[word_embeddings], load_models=True)
    st.write('Done!')
    
    # check if new metrics files have been added
    new_metrics_files = []
    for root, dirs, files in os.walk('./src/'):
        for file in files:
            if file.endswith('.txt'):
                if os.path.join(root, file) not in metrics_files:
                    new_metrics_files.append(os.path.join(root, file))

    # open the new metrics files and display the contents
    if len(new_metrics_files) > 0:
        st.markdown('#### New metrics files:')
        for file in new_metrics_files:
            with open(file, 'r') as f:
                st.markdown(f'##### {file}')
                st.markdown(f'```{f.read()}```')