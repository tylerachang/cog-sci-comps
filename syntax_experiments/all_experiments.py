"""
Run all syntax experiments.
"""

from experiment import *
import os.path

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory')
    parser.add_argument('--num_epochs', type=int)
    return parser

def main(directory, num_epochs):
    
    nmt_models = ['zh-decay-model_step_147000', 'ru-decay-model_step_149000', 'fr-decay-model_step_147000',
           'es-decay-model_step_117000', 'en-decay-model_step_150000', 'ar-decay-model_step_149000']
    
    layers = [3]

    train_suffix = '-conll_dev'
    dev_suffix = '-conll_train-filter8'
    test_suffix = '-conll_test'
    
    train_tags = 'drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/phrase_tags_conll_dev.txt'
    dev_tags = 'drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/phrase_tags_conll_train-filter8.txt'
    test_tags = 'drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/phrase_tags_conll_test.txt'
    
    dev_sentences_path = 'drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/sentences_conll_train-filter8.txt'
    test_sentences_path = 'drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/sentences_conll_test.txt'
    
    for i in range(1, 5):
        for nmt_model in nmt_models:
            output_observations = '{0}/observations-{1}-tag{2}.txt'.format(directory, nmt_model, prediction_tag)
            save_model_path = '{0}/syntax-{1}-tag{2}.pickle'.format(directory, nmt_model, prediction_tag)
            reps_path = 'drive/My Drive/Cog Sci Comps/Sentence Representations/sentence_reps-' + nmt_model
            train_reps = reps_path + train_suffix + '.pt'
            dev_reps = reps_path + dev_suffix + '.pt'
            test_reps = reps_path + test_suffix + '.pt'
            prediction_tag = i
            
            if os.path. exists(output_observations):
                # Already trained this model for this experiment.
                evaluate_model_with_paths(save_model_path, test_reps, test_tags, test_sentences_path,
                                          prediction_tag, layers):
                continue
            
            print('RUNNING MODEL {0} FOR TAG {1}'.format(nmt_model, prediction_tag))
            run_experiment(train_reps, dev_reps, test_reps,
                           train_tags, dev_tags, test_tags, dev_sentences_path, test_sentences_path,
                           save_model_path, output_observations, num_epochs, prediction_tag, layers)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args.directory, args.num_epochs)