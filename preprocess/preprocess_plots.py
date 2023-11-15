import pandas as pd
import re
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordsegment import load, segment
import multiprocessing

df = pd.read_csv('data/filtered_plots_and_genres.csv')
X = df['plot']

load()

def preprocess(text):
        """
        Preprocesses the text
        """    
        text = text.lower()
        text = re.sub(r"[^a-zA-Z;.]+", ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'([;.])', r' \1 ', text)
        tokens = word_tokenize(text)
        
        filtered_tokens = []
        for word in tokens:
            sep = segment(word)
            if len(sep) > 1:
                filtered_tokens.extend(sep)
            else:
                filtered_tokens.append(word)
        tokens = filtered_tokens
        tokens = [word for word in tokens if word not in (stopwords.words('english') + ['.', ';'])]
        tokens = ['<EOS>' if token in ['.', ';'] else token for token in tokens]
        
        preprocessed_text = ' '.join(tokens)

        return preprocessed_text, tokens

def preprocess_parallel(args):
    text = args[0]
    return preprocess(text)

def preprocess_data(data):

    num_processes = multiprocessing.cpu_count() -1
    print(f"Running {num_processes} processes in parallel")

    pool = multiprocessing.Pool(processes=num_processes)

    result_list = list(tqdm(pool.imap(preprocess_parallel, [(text,) for text in data]), total=len(data)))

    pool.close()
    pool.join()

    return zip(*result_list)

X_filtered, _ = preprocess_data(X)

df['plot'] = X_filtered

if __name__ == '__main__':
    df = pd.read_csv('data/filtered_plots_and_genres.csv')
    X = df['plot']
    load()
    X_filtered, _ = preprocess_data(X)
    df['plot'] = X_filtered
    df.to_csv('data/preprocessed_data.csv', index=False)