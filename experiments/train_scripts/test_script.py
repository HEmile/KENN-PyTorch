import os
import sys

import torch.random

sys.path.insert(0, os.getcwd())

import numpy as np
from experiments.preprocessing import generate_dataset
import experiments.train_scripts.train_baseline as tb
import experiments.train_scripts.train_inductive as t  # TODO: change in train_lrl
import pickle
import experiments.settings as s
import os

from kenn.boost_functions import GodelBoostConormApprox, LukasiewiczBoostConorm, ProductBoostConorm

def run_tests(
        n_runs,
        save_results=True,
        custom_training_dimensions=False,
        verbose=True,
        random_seed=s.RANDOM_SEED):

    # SET RANDOM SEED for tensorflow and numpy
    torch.random.manual_seed(random_seed)
    np.random.seed(random_seed)

    training_dimensions = []
    if not custom_training_dimensions:
        print("No custom training dimensions found.")
        training_dimensions = [0.1, 0.25, 0.5, 0.75, 0.9]
        print("Using default training dimensions: {}".format(training_dimensions))
    else:
        training_dimensions = custom_training_dimensions

    results_e2e = {}
    results_greedy = {}

    for td in training_dimensions:
        td_string = str(int(td * 100))
        print(' ========= Start training (' + td_string + '%)  =========')
        results_e2e.setdefault(td_string, {})
        results_greedy.setdefault(td_string, {})

        # results e2e will be dictionaries like this:
        # {'10' : {'NN': [list of n_runs dictionaries containing all the stats], 'KENN': [same as before]},
        #  '25' : {'NN': [list of n_runs dictionaries containing all the stats], 'KENN': [same as before]},
        #   ...}

        for i in range(n_runs):
            print('Generate new dataset: iteration number ' + str(i), flush=True)
            generate_dataset(td, verbose=False)


            print("--- Starting Base NN Training ---")
            results_e2e[td_string].setdefault('NN', []).append(
                tb.train_and_evaluate_standard(td, verbose=verbose)[3])
            print("--- Starting KENN Training - Godel ---")
            results_e2e[td_string].setdefault('Godel', []).append(
                t.train_and_evaluate_kenn_inductive(td,
                                                    boost_function=GodelBoostConormApprox,
                                                    use_preactivations=True,
                                                    verbose=verbose))
            print("--- Starting KENN Training - Lukasiewicz ---")
            results_e2e[td_string].setdefault('Lukasiewicz', []).append(
                t.train_and_evaluate_kenn_inductive(td,
                                                    boost_function=LukasiewiczBoostConorm,
                                                    use_preactivations=False,
                                                    verbose=verbose))
            print("--- Starting KENN Training - Product ---")
            results_e2e[td_string].setdefault('Product', []).append(
                t.train_and_evaluate_kenn_inductive(td,
                                                    boost_function=ProductBoostConorm,
                                                    use_preactivations=True,
                                                    verbose=verbose))


        if save_results:
            with open('./results/results_inductive_{}runs'.format(n_runs), 'wb') as output:
                pickle.dump(results_e2e, output)

    return (results_e2e, results_greedy)


if __name__ == '__main__':
    directory = "./results"

    if not os.path.exists(directory):
        os.makedirs(directory)

    run_tests(n_runs=100, custom_training_dimensions=[0.10, 0.25, 0.50, 0.75, 0.90], verbose=False)