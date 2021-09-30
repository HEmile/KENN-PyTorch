import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.stats import t, norm
sns.set_theme('notebook')
sns.set_style('whitegrid')


def plot_losses(x):
    fig, axes = plt.subplots(len(list(x.keys())), 2)
    for i, key in enumerate(list(x.keys())):
        for j in range(len(x[key]['NN'])):
            if j == 0:
                axes[i, 0].plot(x[key]['NN'][j]['train_losses'],
                                c='b', alpha=0.3, label='Train')
                axes[i, 0].plot(x[key]['NN'][j]['valid_losses'],
                                c='r', alpha=0.4, label='Validation')

            axes[i, 0].plot(x[key]['NN'][j]['train_losses'], c='b', alpha=0.3)
            axes[i, 0].plot(x[key]['NN'][j]['valid_losses'], c='r', alpha=0.4)
            axes[i, 0].legend(loc='best')
            axes[i, 0].set_title('NN ({}%)'.format(key))
            axes[i, 0].set_xlabel('Epochs')
            axes[i, 0].set_ylabel('Loss')

        for j in range(len(x[key]['KENN'])):
            if j == 0:
                axes[i, 1].plot(x[key]['KENN'][j]['train_losses'],
                                c='b', alpha=0.3, label='Train')
                axes[i, 1].plot(x[key]['KENN'][j]['valid_losses'],
                                c='r', alpha=0.4, label='Validation')

            axes[i, 1].plot(x[key]['KENN'][j]['train_losses'],
                            c='b', alpha=0.3)
            axes[i, 1].plot(x[key]['KENN'][j]['valid_losses'],
                            c='r', alpha=0.4)
            axes[i, 1].legend(loc='best')
            axes[i, 1].set_title('KENN ({}%)'.format(key))
            axes[i, 1].set_xlabel('Epochs')
            axes[i, 1].set_ylabel('Loss')

    fig.set_figheight(24)
    fig.set_figwidth(12)
    plt.subplots_adjust(hspace=0.3)

    plt.show()


def plot_accuracies(x):
    fig, axes = plt.subplots(len(list(x.keys())), 2)
    for i, key in enumerate(list(x.keys())):
        for j in range(len(x[key]['NN'])):
            if j == 0:
                axes[i, 0].plot(x[key]['NN'][j]['train_accuracies'],
                                c='b', alpha=0.3, label='Train')
                axes[i, 0].plot(x[key]['NN'][j]['valid_accuracies'],
                                c='r', alpha=0.4, label='Validation')

            axes[i, 0].plot(
                x[key]['NN'][j]['train_accuracies'], c='b', alpha=0.3)
            axes[i, 0].plot(
                x[key]['NN'][j]['valid_accuracies'], c='r', alpha=0.4)
            axes[i, 0].legend(loc='best')
            axes[i, 0].set_title('NN ({}%)'.format(key))
            axes[i, 0].set_xlabel('Epochs')
            axes[i, 0].set_ylabel('Accuracy')

        for j in range(len(x[key]['KENN'])):
            if j == 0:
                axes[i, 1].plot(x[key]['KENN'][j]['train_accuracies'],
                                c='b', alpha=0.3, label='Train')
                axes[i, 1].plot(x[key]['KENN'][j]['valid_accuracies'],
                                c='r', alpha=0.4, label='Validation')

            axes[i, 1].plot(x[key]['KENN'][j]
                            ['train_accuracies'], c='b', alpha=0.3)
            axes[i, 1].plot(x[key]['KENN'][j]
                            ['valid_accuracies'], c='r', alpha=0.4)
            axes[i, 1].legend(loc='best')
            axes[i, 1].set_title('KENN ({}%)'.format(key))
            axes[i, 1].set_xlabel('Epochs')
            axes[i, 1].set_ylabel('Accuracy')

    fig.set_figheight(24)
    fig.set_figwidth(12)
    plt.subplots_adjust(hspace=0.3)

    plt.show()


def get_means_and_stds(history):
    means = []
    stds = []
    means_kenn = []
    stds_kenn = []
    means_deltas = []
    stds_deltas = []

    n_runs = len(history[list(history.keys())[0]]['NN'])

    for num in history.keys():
        test_accuracies = [history[num]['NN'][i]['test_accuracy']
                           for i in range(n_runs)]
        mean_test_accuracies = np.mean(test_accuracies)
        std_test_accuracies = np.std(test_accuracies)

        test_accuracies_kenn = [history[num]['KENN'][i]
                                ['test_accuracy'] for i in range(n_runs)]
        mean_test_accuracies_kenn = np.mean(test_accuracies_kenn)
        std_test_accuracies_kenn = np.std(test_accuracies_kenn)

        deltas = np.array(test_accuracies_kenn) - np.array(test_accuracies)
        mean_deltas = np.mean(deltas)
        std_deltas = np.std(deltas)

        # Append to lists
        means.append(mean_test_accuracies)
        stds.append(std_test_accuracies)
        means_kenn.append(mean_test_accuracies_kenn)
        stds_kenn.append(std_test_accuracies_kenn)
        means_deltas.append(mean_deltas)
        stds_deltas.append(std_deltas)

    return (means, stds, means_kenn, stds_kenn, means_deltas, stds_deltas)


def plot_means_and_intervals(history, title, barwidth=0.3, confidence_level=0.95):

    confidence_margins_nn, confidence_margins_kenn, _ = get_all_confidence_margins(
        history, confidence_level)
    means, stds, means_kenn, stds_kenn, _, _ = get_means_and_stds(history)

    plt.figure(figsize=(9, 5))
    # Set position of bar on X axis
    r1 = np.arange(len(means))
    r2 = [x + barwidth for x in r1]

    # Make the plot
    plt.bar(r1, means, color='b', width=barwidth,
            edgecolor='white', label='NN')
    plt.bar(r2, means_kenn, color='r', width=barwidth,
            edgecolor='white', label='KENN')
    plt.errorbar(
        r1,
        means,
        yerr=confidence_margins_nn,
        capsize=5,
        color='black',
        elinewidth=2,
        ls='none')
    plt.errorbar(
        r2,
        means_kenn,
        yerr=confidence_margins_kenn,
        capsize=5,
        color='black',
        elinewidth=2,
        ls='none')

    # Add xticks on the middle of the group bars
    plt.xlabel('Percentage of Training', fontweight='bold')
    plt.ylabel('Test Accuracy', fontweight='bold')
    plt.xticks([r + barwidth for r in range(len(means))], history.keys())
    plt.legend(loc='best')
    plt.title(title)

    plt.savefig('plots/' + title + '.png')
    plt.show()


def plot_deltas(history, barwidth=0.3, title='', other_deltas='', confidence_level=0.95):
    assert(other_deltas == '' or other_deltas == 'i' or other_deltas == 't')
    _, _, _, _, means_deltas, stds_deltas = get_means_and_stds(history)

    _, _, confidence_margins_delta = get_all_confidence_margins(
        history, confidence_level=confidence_level)

    results_NN_Marra_i = np.array([0.645, 0.674, 0.707, 0.717, 0.723])
    results_SBR_i = np.array([0.650, 0.682, 0.712, 0.719, 0.726])
    results_RNM_i = np.array([0.685, 0.709, 0.726, 0.726, 0.732])

    results_NN_Marra_t = np.array([0.640, 0.667, 0.695, 0.708, 0.726])
    results_SBR_t = np.array([0.703, 0.729, 0.747, 0.764, 0.780])
    results_RNM_t = np.array([0.708, 0.735, 0.753, 0.766, 0.780])

    deltas_SBR_i = results_SBR_i - results_NN_Marra_i
    deltas_RNM_i = results_RNM_i - results_NN_Marra_i

    deltas_SBR_t = results_SBR_t - results_NN_Marra_t
    deltas_RNM_t = results_RNM_t - results_NN_Marra_t

    deltas = means_deltas
    r = np.arange(len(deltas))

    plt.figure(figsize=(9, 5))
    plt.bar(r, deltas, color='b', width=barwidth,
            edgecolor='white', label='delta KENN')
    plt.errorbar(
        r,
        deltas,
        yerr=confidence_margins_delta,
        capsize=5,
        color='black',
        elinewidth=2,
        ls='none')

    if other_deltas == 'i':
        r2 = [x + barwidth for x in r]
        r3 = [x + barwidth for x in r2]
        plt.bar(r2, deltas_SBR_i, color='r', width=barwidth,
                edgecolor='white', label='delta SBR')
        plt.bar(r3, deltas_RNM_i, color='g', width=barwidth,
                edgecolor='white', label='delta RNM')

    elif other_deltas == 't':
        r2 = [x + barwidth for x in r]
        r3 = [x + barwidth for x in r2]
        plt.bar(r2, deltas_SBR_t, color='r', width=barwidth,
                edgecolor='white', label='delta SBR')
        plt.bar(r3, deltas_RNM_t, color='g', width=barwidth,
                edgecolor='white', label='delta RNM')

    plt.xlabel('Percentage of Training', fontweight='bold')
    plt.xticks([r + barwidth for r in range(len(means_deltas))],
               list(history.keys()))
    plt.legend(loc='best')
    plt.title(title)
    plt.savefig('plots/' + 'deltas_' + title + '.png')
    plt.show()


def plot_histograms(history, title, bw=0.5, bins=20):
    n_runs = len(history[list(history.keys())[0]]['NN'])

    fig, axes = plt.subplots(len(history.keys()), 2, figsize=(12, 13))

    for i, num in enumerate(history.keys()):

        test_accuracies = [history[num]['NN'][i]
                           ['test_accuracy'].numpy() for i in range(n_runs)]
        test_accuracies_kenn = [history[num]['KENN'][i]
                                ['test_accuracy'].numpy() for i in range(n_runs)]
        deltas = list(np.array(test_accuracies_kenn) -
                      np.array(test_accuracies))

        # Draw histograms
        b = np.histogram(np.hstack((test_accuracies, test_accuracies_kenn)), bins=bins)[
            1]  # get the bin edges

        axes[i, 0].hist(test_accuracies, density=True, bins=b,
                        alpha=0.3, label='NN', color='blue')
        sns.kdeplot(test_accuracies, ax=axes[i, 0], bw_adjust=bw, color='blue')

        axes[i, 0].hist(test_accuracies_kenn, density=True,
                        bins=b, alpha=0.3, label='KENN', color='red')
        sns.kdeplot(test_accuracies_kenn,
                    ax=axes[i, 0], bw_adjust=bw, color='red')

        axes[i, 0].set_title("Training dimension: {}".format(num))
        axes[i, 0].legend(loc='best')

        axes[i, 1].hist(deltas, density=True, bins=bins,
                        alpha=0.3, label='KENN', color='green')
        sns.kdeplot(deltas, ax=axes[i, 1], bw_adjust=bw, color='green')
        axes[i, 1].set_title("Deltas for training dimension {}%".format(num))

    fig.tight_layout()
    plt.savefig('plots/' + title + '.png')
    plt.show()


def print_stats(history):
    means, stds, means_kenn, stds_kenn, means_deltas, stds_deltas = get_means_and_stds(
        history)
    p_vals = make_t_test(history)
    for i, key in enumerate(history.keys()):
        print("== {}% ==".format(key))
        print("Mean Test Accuracy:\tNN = {:8.6f}; KENN = {:8.6f}".format(
            means[i], means_kenn[i]))
        print("Test Accuracy std:\tNN = {:8.6f}; KENN = {:8.6f}".format(
            stds[i], stds_kenn[i]))
        print("\t\t\tDeltas Mean = {:8.6f}".format(means_deltas[i]))
        print("\t\t\tDeltas Std = {:8.6f}".format(stds_deltas[i]))
        print("\t\t\tright tailed p-value: {}".format(p_vals[i]))
        print()


def print_and_plot_results(history, plot_title, other_deltas='', confidence_level=0.95):
    """
    Parameters:
    - other_deltas: a string taking values in ['', 'i', 't'].
        - '': Only the deltas from kenn are plotted:
        - 'i': The deltas from the other inductive experiments are printed along our deltas
        - 't': The deltas from the other transductive experiments are printed along our deltas
    """
    # means, stds, means_kenn, stds_kenn = get_means_and_stds(history)
    print_stats(history)
    plot_means_and_intervals(history, plot_title, 0.4,
                             confidence_level=confidence_level)
    plot_deltas(history, title=plot_title, other_deltas=other_deltas)


def plot_clause_weights(history):

    topics = ["AI", "Agents", "DB", "HCI", "IR", "ML"]

    fig, axes = plt.subplots(1, 3)

    for i, l in enumerate(history['clause_weights']):
        for j in range(len(np.transpose(l))):
            axes[i].plot(np.transpose(l)[j], label=topics[j])
        axes[i].set_title('KENN Layer {}'.format(i+1))
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel('Value')
        axes[i].legend(loc='best')

    fig.set_figheight(5)
    fig.set_figwidth(15)
    plt.subplots_adjust(hspace=0.3)
    plt.savefig('plots/clause_weights.png')


def make_t_test(history):
    p_values = []
    n_runs = len(history[list(history.keys())[0]]['NN'])
    for i, num in enumerate(history.keys()):
        test_accuracies = [history[num]['NN'][i]
                           ['test_accuracy'].numpy() for i in range(n_runs)]
        test_accuracies_kenn = [history[num]['KENN'][i]
                                ['test_accuracy'].numpy() for i in range(n_runs)]

        p_values.append(scipy.stats.ttest_ind(test_accuracies_kenn,
                        test_accuracies, alternative='greater', equal_var=False)[1])
    return p_values


def get_confidence_margin(samples, confidence_level):
    alpha = 1-confidence_level
    n = len(samples)
    df = n-1
    t1 = t.ppf(1-alpha/2, df=df)
    cm = (t1*(np.std(samples)/np.sqrt(n)))
    return cm


def get_all_confidence_margins(history, confidence_level):
    confidence_margins_nn = []
    confidence_margins_kenn = []
    confidence_margins_deltas = []

    # compute confidence margins
    n_runs = len(history[list(history.keys())[0]]['NN'])
    for i, num in enumerate(history.keys()):
        test_accuracies = [history[num]['NN'][i]
                           ['test_accuracy'].numpy() for i in range(n_runs)]
        test_accuracies_kenn = [history[num]['KENN'][i]
                                ['test_accuracy'].numpy() for i in range(n_runs)]
        deltas = list(np.array(test_accuracies_kenn) -
                      np.array(test_accuracies))
        confidence_margins_nn.append(
            get_confidence_margin(test_accuracies, confidence_level))
        confidence_margins_kenn.append(get_confidence_margin(
            test_accuracies_kenn, confidence_level))
        confidence_margins_deltas.append(
            get_confidence_margin(deltas, confidence_level))

    return confidence_margins_nn, confidence_margins_kenn, confidence_margins_deltas
