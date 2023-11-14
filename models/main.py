from xgb_w2v import XGBRunner

if __name__ == '__main__':
    runner = XGBRunner()
    runner.run_inference()
    runner.run_training()