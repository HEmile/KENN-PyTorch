import pandas as pd
import numpy as np
import experiments.settings as s


def unary_index(ids_list, id):
    """
    Returns the index of the id inside ids_list. If multiple indexes are found, throws an error,
    and if no index is found, raises an Exception. If only one is found, returns it.

    Parameters:
    - ids_list = the list of all the ids;
    - id = the specific id we want to know the index of."""
    match = np.where(ids_list == id)[0]

    assert len(match) < 2

    if len(match) == 0:
        raise Exception(id)
    else:
        return match[0]


def permute(y, samples_per_class: int):
    '''Return a permutation of the dataset indexes, with samples_per_class samples for each class in the first rows.
    The first rows are used outside as a training set (which must be balanced)
    Parameters:
    - y = the pd.Series containing all the classes of all the samples
    - samples_per_class = int representing the number of samples we want to keep for each class when
      balancing the dataset.
    '''

    # List of classes
    classes = list(set(y))

    # initialize empty arrays, will be filled below
    p_train = np.array([], dtype=np.int32)
    p_valid = np.array([], dtype=np.int32)
    p_other = np.array([], dtype=np.int32)

    for c in classes:
        # the [0] is because np.where returns a tuple with an array inside, and we want the array
        # i_c contains the indexes of samples in the dataset of class c
        i_c = np.where(y == c)[0]
        # generate a shuffled series of indexes
        p = np.random.permutation(len(i_c))
        # shuffle i_c according to the shuffled indexes we generated
        i_c = i_c[p]

        # Here we fill up the empty arrays initialized above (with np.concatenate).
        # Here we select only the desired amount of samples for each class
        p_train = np.concatenate((p_train, i_c[:samples_per_class]), axis=0)
        p_other = np.concatenate((p_other, i_c[samples_per_class:]), axis=0)

    p_train = np.random.permutation(p_train)

    return np.concatenate((p_train, np.random.permutation(p_other)), axis=0)


def generate_indexes_inductive(relations, ids_list, samples_in_training, samples_in_valid, verbose=True):
    '''Generate the indexes to be used by kenn for the relational part for the Inductive learning task
    Specifically, this function returns the index couples (s1,s2) of all the edges for the inductive case:
    i.e. we remove edges (n1,n2) s.t. n1 is in the training set, and n2 is in the test set.

    Parameters:
    :param relations: array containing the edges of the graph. The format is [cited_paper, citing paper]
    :param ids_list: array containing the ids of all the samples
    :param samples_in_training: n. of samples in the training set. Needed to tell if an index refers to a
    sample of the training set or of the test set.
    '''
    s1_training = []
    s2_training = []
    s1_valid = []
    s2_valid = []
    s1_test = []
    s2_test = []

    # iterate over the length of the number of edges
    for i in range(len(relations)):
        try:
            # here we fetch the index of the FIRST element of the i-th edge
            match1 = unary_index(ids_list, relations[i, 0])
            # here we fetch the index of the SECOND element of the i-th edge
            match2 = unary_index(ids_list, relations[i, 1])

            # Here we decide if we want to put the found indexes in s*_training or in s*_test or s*_valid
            if match1 < samples_in_training and match2 < samples_in_training:
                # Both in training set
                s1_training.append([match1])
                s2_training.append([match2])
                # both in validation set
            elif match1 in range(samples_in_training, samples_in_training + samples_in_valid) and match2 in range(
                    samples_in_training, samples_in_training + samples_in_valid):
                s1_valid.append([match1 - samples_in_training])
                s2_valid.append([match2 - samples_in_training])
            elif match1 >= (samples_in_training + samples_in_valid) and match2 >= (
                    samples_in_training + samples_in_valid):
                # Both in test set
                s1_test.append([match1 - (samples_in_training + samples_in_valid)])
                s2_test.append([match2 - (samples_in_training + samples_in_valid)])
        except Exception as e:
            if verbose:
                print('Missing paper, removed citation! Id: ')
                print(e)

    return np.array(s1_training).astype(np.int32), np.array(s2_training).astype(np.int32), \
           np.array(s1_valid).astype(np.int32), np.array(s2_valid).astype(np.int32), \
           np.array(s1_test).astype(np.int32), np.array(s2_test).astype(np.int32)


def generate_indexes_transductive(relations, ids_list, verbose=True):
    '''Generate the indexes to be used by kenn for the relational part, for the Transductive learning task.
    Specifically, this function returns the index couples (s1,s2) of all the edges for the transductive case:
    i.e. we don't remove edges (n1,n2) s.t. n1 is in the training set, and n2 is in the test set.

    Parameters:
    - relations: np.array containing the edges of the graph. The format is [cited_paper, citing paper];
    - ids_list: np.array containing the ids of all the samples'''
    s1 = []
    s2 = []

    for i in range(len(relations)):
        try:
            # here we fetch the index of the FIRST element of the i-th edge
            match1 = unary_index(ids_list, relations[i, 0])
            # here we fetch the index of the SECOND element of the i-th edge
            match2 = unary_index(ids_list, relations[i, 1])

            s1.append([match1])
            s2.append([match2])
        except Exception as e:
            if verbose:
                print('Missing paper, removed citation! Id: ')
                print(e)
    # we return s1 and s2 as np.arrays and as column vectors (shape: (len(relations),1))
    return np.array(s1).astype(np.int32), np.array(s2).astype(np.int32)


def get_train_and_valid_lengths(features, percentage_of_training):
    """
    Given the array with the whole dataset, and the percentage of the training set,
    returns the lengths of the training set and validation set, such that the training set
    is balanced and that the validation set is s.VALIDATION_PERCENTAGE * train_length
    """
    total_number_of_samples = features.shape[0]
    # t_all = len(X_train + X_val)
    t_all = int(round(total_number_of_samples * percentage_of_training))
    number_of_samples_training = int(round(t_all * (1. - s.VALIDATION_PERCENTAGE)))

    # Since we want a balanced training dataset (i.e. same number of samples of the same class, for each class)
    # we define how many samples per class we want.
    samples_per_class = int(round(number_of_samples_training / s.NUMBER_OF_CLASSES))

    number_of_samples_training = samples_per_class * s.NUMBER_OF_CLASSES
    number_of_samples_validation = t_all - number_of_samples_training
    return number_of_samples_training, number_of_samples_validation


def generate_dataset(percentage_of_training, verbose=True):
    """
    Generates dataset files in .npy format.
    Attributes:
    - percentage_of_training: number between 0 and 1. Fraction of the training set to use.
    By default it's set to 1, so the full dataset is used."""

    dataset = pd.read_csv(s.DATASET_FOLDER + s.DATASET + '.content', delimiter='\t', header=None)

    # First column contains indexes, last one the label. In the middle, the features
    # ids: id of each paper (shape: (3312,1));
    # x: the features for each paper (shape:(3312, 3703));
    # y: the class for each paper (shape:(3312,1)).
    ids, x, y = np.split(dataset, [1, -1], axis=1)

    number_of_samples_training, number_of_samples_validation = get_train_and_valid_lengths(x, percentage_of_training)

    p = permute(y.iloc[:, 0], int(number_of_samples_training // s.NUMBER_OF_CLASSES))

    # balances and shuffles ids and x
    ids = ids.to_numpy().astype(np.str)[p, :]
    x = x.to_numpy().astype(np.float32)[p, :]

    y = pd.get_dummies(y, prefix=['class'])
    # just gets a list with the names of all the possible classes
    classes = [c[5:] for c in list(y.columns)]

    # balances and shuffles y
    y = y.to_numpy().astype(np.float32)[p, :]

    # read the Graph data from citeseer.cites and put them in a np.array
    relations = pd.read_csv(s.DATASET_FOLDER + s.DATASET + '.cites', delimiter='\t', header=None).to_numpy()

    # s1 and s2 will be all couples of indexes refering to edges of the graph.
    # transductive means that no edge is removed even if both endnodes are in different sets
    s1, s2 = generate_indexes_transductive(relations, ids, verbose=verbose)
    # b will be the "preactivations" of the binary clauses that we give as input to the KE.
    # since we consider only couple of nodes that are actually connected, we generate an array
    # of ones of length len(relations).
    # Also, since those are preactivations, we want to pick a value z such that sigmoid(z) ~ 1.
    # since logit(1) is not defined, we just pick logit(1-epsilon) which is some high number. We pick 500.
    b = np.ones([len(s1), 1], dtype=np.float32) * 500

    # same as s1 and s2, but now we separate training and test set (i.e. remove all edges (n1,n2) where
    # n1 is inside training set and n2 is inside test set.
    s1tr, s2tr, s1val, s2val, s1te, s2te = generate_indexes_inductive(relations, ids, number_of_samples_training,
                                                                      samples_in_valid=number_of_samples_validation,
                                                                      verbose=verbose)

    # We do the same thing for the binary predicates preactivations
    btr = np.ones([len(s1tr), 1], dtype=np.float32) * 500
    bval = np.ones([len(s1val), 1], dtype=np.float32) * 500
    bte = np.ones([len(s1te), 1], dtype=np.float32) * 500

    np.save(s.DATASET_FOLDER + 'features', x)
    np.save(s.DATASET_FOLDER + 'labels', y)
    np.save(s.DATASET_FOLDER + 'relations_transductive', b)

    np.save(s.DATASET_FOLDER + 'index_x_transductive', s1)
    np.save(s.DATASET_FOLDER + 'index_y_transductive', s2)

    np.save(s.DATASET_FOLDER + 'index_x_inductive_training', s1tr)
    np.save(s.DATASET_FOLDER + 'index_y_inductive_training', s2tr)
    np.save(s.DATASET_FOLDER + 'relations_inductive_training', btr)
    np.save(s.DATASET_FOLDER + 'index_x_inductive_validation', s1val)
    np.save(s.DATASET_FOLDER + 'index_y_inductive_validation', s2val)
    np.save(s.DATASET_FOLDER + 'relations_inductive_validation', bval)
    np.save(s.DATASET_FOLDER + 'index_x_inductive_test', s1te)
    np.save(s.DATASET_FOLDER + 'index_y_inductive_test', s2te)
    np.save(s.DATASET_FOLDER + 'relations_inductive_test', bte)

    # Generate knowledge
    kb = ''

    # List of predicates
    for c in classes:
        kb += c + ','

    kb = kb[:-1] + '\nCite\n\n'

    # No unary clauses

    kb = kb[:-1] + '\n>\n'

    # Binary clauses

    # nC(x),nCite(x.y),C(y)
    for c in classes:
        kb += '_:n' + c + '(x),nCite(x.y),' + c + '(y)\n'

    with open('knowledge_base', 'w') as kb_file:
        kb_file.write(kb)


if __name__ == '__main__':
    generate_dataset(0.9)
