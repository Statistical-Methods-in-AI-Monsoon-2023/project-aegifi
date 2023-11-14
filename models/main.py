from xgb_w2v import XGBRunner

if __name__ == '__main__':
    runner = XGBRunner(word_embeddings='TF-IDF')
    runner.run_training(save_model=True)
    # runner.run_inference()