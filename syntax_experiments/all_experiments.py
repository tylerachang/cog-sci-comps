"""
Run all syntax experiments.
"""

from experiment import *
import os.path

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory')
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--evaluation', type=bool)
    return parser

def main(directory, num_epochs, is_evaluation):
    
    nmt_models = ['zh-decay-model_step_147000', 'ru-decay-model_step_149000', 'fr-decay-model_step_147000',
           'es-decay-model_step_117000', 'en-decay-model_step_150000', 'ar-decay-model_step_149000']
    
    layers = [3,7]

    train_suffix = '-conll_dev'
    dev_suffix = '-conll_train-filter8'
    test_suffix = '-conll_test'
    
    train_tags = 'drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/phrase_tags_conll_dev.txt'
    dev_tags = 'drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/phrase_tags_conll_train-filter8.txt'
    test_tags = 'drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/phrase_tags_conll_test.txt'
    
    dev_sentences_path = 'drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/sentences_conll_train-filter8.txt'
    test_sentences_path = 'drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/sentences_conll_test.txt'
    
    for nmt_model in nmt_models:
        reps_path = 'drive/My Drive/Cog Sci Comps/Sentence Representations/sentence_reps-' + nmt_model
        train_reps = reps_path + train_suffix + '.pt'
        dev_reps = reps_path + dev_suffix + '.pt'
        test_reps = reps_path + test_suffix + '.pt'
        if is_evaluation:
            X_train = None
            X_dev = None
        else:
            X_train = load_reps(train_reps, layers)
            X_dev = load_reps(dev_reps, layers)
        X_test = load_reps(test_reps, layers)
        
        for i in range(1, 5):
            prediction_tag = i
            output_observations = '{0}/observations-{1}-tag{2}.txt'.format(directory, nmt_model, prediction_tag)
            output_predictions = '{0}/predictions-{1}-tag{2}.txt'.format(directory, nmt_model, prediction_tag)
            save_model_path = '{0}/syntax-{1}-tag{2}.pickle'.format(directory, nmt_model, prediction_tag)
            
            if os.path. exists(output_observations):
                # Already trained this model for this experiment.
                # Can test the existing models:
                if is_evaluation:
                    print('TESTING EXISTING MODEL {0} FOR TAG {1}'.format(nmt_model, prediction_tag))
                    evaluate_model_with_paths(save_model_path, X_test, test_tags, test_sentences_path,
                                              prediction_tag, layers)
                continue
            
            print('RUNNING MODEL {0} FOR TAG {1}'.format(nmt_model, prediction_tag))
            # Passes in X_train, X_dev, and X_test directly to avoid having to load multiple times.
            # Could also pass in the paths instead (train_reps, dev_reps, test_reps).
            run_experiment(X_train, X_dev, X_test,
                           train_tags, dev_tags, test_tags, dev_sentences_path, test_sentences_path,
                           save_model_path, output_observations, output_predictions, num_epochs,
                           prediction_tag, layers)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args.directory, args.num_epochs, args.evaluation)