from xgb_w2v import XGBRunner
from binary_nb import BinaryNBRunner
from multinomial_nb import MultinomailNBRunner
class Model:
    def __init__(self, model='xgb', word_embeddings='w2v', load_models=False):
        if model == 'xgb':
            self.model = XGBRunner(word_embeddings=word_embeddings, load_models=load_models)
        elif model == 'bnb':
            self.model = BinaryNBRunner(load_models=load_models)
        elif model == 'mnb':
            self.model = MultinomailNBRunner(load_models=load_models)
        else:
            raise Exception('Invalid model name')

if __name__ == '__main__':
    md = Model(model='xgb', word_embeddings='w2v', load_models=False)
    md.model.run_training(save_model=True)
    # runner.run_inference()