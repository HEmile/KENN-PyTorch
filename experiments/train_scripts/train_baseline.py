from experiments.model import Standard
import numpy as np
import experiments.settings as s
from experiments.preprocessing import *
from experiments.util import *


def train_and_evaluate_standard(percentage_of_training, verbose=True):
    """
    Trains Standard model with the Training Set, validates on Validation Set
    and evaluates accuracy on the Test Set.
    """
    standard_model = Standard(s.NUMBER_OF_FEATURES)

    optimizer = torch.optim.Adam(standard_model.parameters())

    # LOADING DATASET
    features = np.load(s.DATASET_FOLDER + 'features.npy')
    labels = np.load(s.DATASET_FOLDER + 'labels.npy')

    train_len, samples_in_valid = get_train_and_valid_lengths(
        features, percentage_of_training)

    features = torch.tensor(features)
    labels = torch.tensor(labels)

    train_losses = []
    valid_losses = []
    valid_accuracies = []
    train_accuracies = []

    train_indices = range(train_len)
    valid_indices = range(train_len, train_len + samples_in_valid)
    test_indices = range(train_len + samples_in_valid, features.shape[0])

    # TRAIN AND EVALUATE STANDARD MODEL
    for epoch in range(s.EPOCHS):
        train_step_standard(
            model=standard_model,
            features=features[train_indices, :],
            labels=labels[train_indices, :],
            optimizer=optimizer
        )

        _, t_predictions = standard_model(features[train_indices, :])
        t_loss = loss(t_predictions, labels[train_indices, :])

        v_predictions, v_loss = validation_step_standard(
            model=standard_model,
            features=features[valid_indices, :],
            labels=labels[valid_indices, :],
        )

        train_losses.append(t_loss)
        valid_losses.append(v_loss)

        t_accuracy = accuracy(t_predictions, labels[train_indices, :])
        v_accuracy = accuracy(v_predictions, labels[valid_indices, :])

        train_accuracies.append(t_accuracy)
        valid_accuracies.append(v_accuracy)

        if verbose and epoch % 10 == 0:
            print(
                "Epoch {}: Training Loss: {:5.4f} Validation Loss: {:5.4f} | Train Accuracy: {:5.4f} Validation Accuracy: {:5.4f};".format(
                    epoch, t_loss, v_loss, t_accuracy, v_accuracy))

        # Early Stopping
        stopEarly = callback_early_stopping(valid_accuracies)
        if stopEarly:
            print("callback_early_stopping signal received at epoch= %d/%d" %
                  (epoch, s.EPOCHS))
            print("Terminating training ")
            break

    preactivations_train, _ = standard_model(features[train_indices, :])
    preactivations_valid, _ = standard_model(features[valid_indices, :])
    preactivations_test, predictions_test = standard_model(
        features[test_indices, :])
    test_accuracy = accuracy(predictions_test, labels[test_indices, :])
    print("Test Accuracy: {}".format(test_accuracy))

    nn_results = {"train_losses": train_losses,
                  "train_accuracies": train_accuracies,
                  "valid_losses": valid_losses,
                  "valid_accuracies": valid_accuracies,
                  "test_accuracy": test_accuracy}

    return (
        preactivations_train,
        preactivations_valid,
        preactivations_test,
        nn_results
    )