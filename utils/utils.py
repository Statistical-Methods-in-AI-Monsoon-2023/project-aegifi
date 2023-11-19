# utils file
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from nltk.tokenize import word_tokenize

def load_w2v():
    
    X = np.load('vectorised_data/X_w2v.npy')
    y = np.load('vectorised_data/y.npy')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print('Loaded data')
    
    return X_train, X_test, y_train, y_test

def load_tf_w2v():
    X = np.load('vectorised_data/X_tfidf_w2v.npy')
    y = np.load('vectorised_data/y.npy')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print('Loaded data')
    
    return X_train, X_test, y_train, y_test

def load_gru():
    X = np.load('vectorised_data/X_gru.npy')
    y = np.load('vectorised_data/y.npy')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print('Loaded data')
    
    return X_train, X_test, y_train, y_test

def load_bow():
    X = scipy.sparse.load_npz('vectorised_data/X_bow.npz')
    y = np.load('vectorised_data/y.npy')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print('Loaded data')
    
    return X_train, X_test, y_train, y_test

def load_tfidf():
    X = scipy.sparse.load_npz('vectorised_data/X_tfidf.npz')
    y = np.load('vectorised_data/y.npy')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print('Loaded data')
    
    return X_train, X_test, y_train, y_test

def load_data(word_embeddings='tfidf'):
    loaders = {
        'gru': load_gru,
        'w2v': load_w2v,
        'bow': load_bow,
        'tf_w2v': load_tf_w2v,
        'tfidf': load_tfidf
    }

    return loaders[word_embeddings]()

def hit_rate(predicted, true):
    # Use element-wise logical AND to check if at least one class is predicted correctly
    correct_predictions = np.any(np.logical_and(predicted, true), axis=1)
    
    # Calculate the hit rate as the percentage of correct samples
    hit_rate = (np.sum(correct_predictions) / len(correct_predictions))
    
    return hit_rate

class MetricReader:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.metrics = {}
        self.metric_types = []
        self.dfs = {}
        self.model_name = {
            'xgb': 'XGBoost',
            'bgru': 'Binary GRU',
            'rgru': 'Rank GRU',
            'mgru': 'Multinomial GRU',
            'bnb': 'Binary Naive Bayes',
            'mnb': 'Multinomial Naive Bayes'
        }
    
    def get_model_name(self, file_path):
        name_path =  file_path.split('/')[-1].split('.')[0]
        if 'xgb' in name_path:
            return 'xgb'
        elif 'binary_gru' in name_path:
            return 'bgru'
        elif 'rank_gru' in name_path:
            return 'rgru'
        elif 'multinomial_gru' in name_path:
            return 'mgru'
        elif 'binary_nb' in name_path:
            return 'bnb'
        elif 'multi_nb' in name_path:
            return 'mnb'
    
    def create_df_for_metric(self, metric_name):
        # create a dataframe for the metric for all models
        df_dict = {
            'Model': [],
            'Value': []
        }
        for model_name in self.metrics:
            df_dict['Model'].append(self.model_name[model_name])
            df_dict['Value'].append(float(self.metrics[model_name][metric_name]))
        # create dataframe from dictionary
        df = pd.DataFrame(df_dict)
        self.dfs[metric_name] = df
    
    def print_metrics(self):
        for model_name in self.metrics:
            print(f'Model: {model_name}')
            for metric_name in self.metrics[model_name]:
                print(f'{metric_name}: {self.metrics[model_name][metric_name]}')
            print()
    
    def read_single_file(self, file_path):
        model_name = self.get_model_name(file_path)
        print(f'Model: {model_name}')
        model_metrics = {}
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # remove the newline character
                line = line.strip()
                metrics= line.split(':')
                metric_name = metrics[0]
                metric_value = metrics[1]
                model_metrics[metric_name] = metric_value
                
                if metric_name not in self.metric_types:
                    self.metric_types.append(metric_name)
                
        self.metrics[model_name] = model_metrics
    
    def create_dfs(self):
        for metric_name in self.metric_types:
            self.create_df_for_metric(metric_name)
    
    def print_dfs(self):
        for metric_name in self.dfs:
            print(f'Metric: {metric_name}')
            print(self.dfs[metric_name])
            print()
    
    def read_files(self):
        for file_path in self.file_paths:
            self.read_single_file(file_path)
        self.create_dfs()
        self.print_dfs()
        return self.dfs

if __name__=='__main__':
    metric_reader = MetricReader(['./src/naive_bayes/metrics/multi_nb_202311200014.txt', './src/naive_bayes/metrics/binary_nb_202311191734.txt'])
    metric_reader.read_files()