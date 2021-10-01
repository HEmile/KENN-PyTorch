import torch
import numpy as np
import experiments.settings as s
from torch.optim import Optimizer
from experiments.model import Standard, Kenn
from torch import Tensor
from numpy.typing import ArrayLike


def accuracy(predictions: Tensor, labels: Tensor) -> Tensor:
    # Accuracy
    correctly_classified = torch.argmax(predictions, 1) == torch.argmax(labels, 1)
    return torch.mean(correctly_classified.float())


def callback_early_stopping(AccList: ArrayLike, min_delta=s.ES_MIN_DELTA, patience=s.ES_PATIENCE) -> bool:
    """
    Takes as argument the list with all the validation accuracies.
    If patience=k, checks if the mean of the last k accuracies is higher than the mean of the
    previous k accuracies (i.e. we check that we are not overfitting). If not, stops learning.
    """
    # No early stopping for 2*patience epochs
    if len(AccList)//patience < 2:
        return False
    # Mean loss for last patience epochs and second-last patience epochs
    mean_previous = np.mean(AccList[::-1][patience:2*patience])
    mean_recent = np.mean(AccList[::-1][:patience])
    delta = mean_recent - mean_previous

    if delta <= min_delta:
        print(
            "*CB_ES* Validation Accuracy didn't increase in the last %d epochs" % (patience))
        print("*CB_ES* delta:", delta)
        return True
    return False

def loss(predictions: Tensor, labels: Tensor) -> Tensor:
    l = torch.nn.NLLLoss()
    return l(predictions, torch.argmax(labels, dim=-1))

def train_step_standard(model: Standard, features: Tensor, labels: Tensor, optimizer: Optimizer):
    """
    Train step for the base NN.
    Parameters:
    - model: the tf.models.Model variable containing the standard NN model we are training;
    - features: the subset of samples on which we are training the model (i.e. Training Set):
    - labels: the ground truth labels corresponding to the samples given in the features parameter;
    - loss: tf.loss object associated with the model;
    - optimizer: tf.optimizer object associated with the model.
    """
    optimizer.zero_grad()
    _, predictions = model(features)
    training_loss = loss(predictions, labels)

    training_loss.backward()
    optimizer.step()


def validation_step_standard(model: Standard, features: Tensor, labels: Tensor) -> (Tensor, Tensor):
    """
    Validation step for the base NN.
    Parameters:
    - model: the tf.models.Model variable containing the standard NN model we are validating;
    - features: the subset of samples on which we are validating the model (i.e. Validation Set):
    - labels: the ground truth labels corresponding to the samples given in the features parameter;
    - loss: tf.loss object associated with the model.
    """
    with torch.no_grad():
        _, predictions = model(features)
        valid_loss = loss(predictions, labels)
        return predictions, valid_loss


def train_step_kenn_inductive(model: Kenn, features: Tensor, relations: Tensor,
                              index_x_train: Tensor, index_y_train: Tensor, labels: Tensor, optimizer: Optimizer):
    """
    Train step for the KENN model, for the Inductive Paradigm
    (i.e. we are using only relations completely inside the training set).
    Parameters:
    - model: the tf.models.Model variable containing the KENN model we are training;
    - features: the subset of samples on which we are training the model (i.e. Training Set);
    - relations: the vector with binary relations preactivations (i.e. all 1 in our case)
    - index_x_train: indexes corresponding to the first objects of the binary relation pairs;
    - index_y_train: indexes corresponding to the second objects of the binary relation pairs;
    (NB: len(index_x_train) == len(index_y_train) == len(relations))
    - labels: the ground truth labels corresponding to the samples given in the features parameter;
    - loss: tf.loss object associated with the model;
    - optimizer: tf.optimizer object associated with the model.
    """

    optimizer.zero_grad()
    predictions_KENN = model([features, relations, index_x_train, index_y_train])

    l = loss(predictions_KENN, labels)

    l.backward()
    optimizer.step()


def train_step_kenn_transductive(model, features, relations, index_x_train, index_y_train, labels, optimizer: Optimizer):
    """
    Train step for the KENN model, for the Transductive Paradigm
    (i.e. we are using only relations completely inside the training set).
    Parameters:
    - model: the tf.models.Model variable containing the KENN model we are training;
    - features: All the samples, since we are in the Transductive case (the actual learning will be performed considering only the loss on the training samples)
    - relations: the vector with binary relations preactivations (i.e. all 1 in our case)
    - index_x_train: indexes corresponding to the first objects of the binary relation pairs;
    - index_y_train: indexes corresponding to the second objects of the binary relation pairs;
    (NB: len(index_x_train) == len(index_y_train) == len(relations))
    - labels: the ground truth labels corresponding to the samples of the Training Set;
    - loss: tf.loss object associated with the model;
    - optimizer: tf.optimizer object associated with the model.
    """
    optimizer.zero_grad()
    predictions_KENN = model([features, torch.tensor(relations, dtype=torch.int64),
                              torch.tensor(index_x_train, dtype=torch.int64),
                              torch.tensor(index_y_train, dtype=torch.int64)])

    l = loss(predictions_KENN[:len(labels), :], labels)
    l.backward()
    optimizer.step()


def validation_step_kenn_inductive(model, features, relations, index_x_valid, index_y_valid, labels):
    """
    Validation step for the KENN model, for the Inductive Paradigm
    (i.e. we are using only relations completely inside the validation set).
    Parameters:
    - model: the tf.models.Model variable containing the KENN model we are validating;
    - features: the subset of samples on which we are validating the model (i.e. Validation Set);
    - relations: the vector with binary relations preactivations (i.e. all 1 in our case)
    - index_x_valid: indexes corresponding to the first objects of the binary relation pairs;
    - index_y_valid: indexes corresponding to the second objects of the binary relation pairs;
    (NB: len(index_x_train) == len(index_y_train) == len(relations))
    - labels: the ground truth labels corresponding to the samples given in the features parameter;
    - loss: tf.loss object associated with the model.
    """
    with torch.no_grad():
        predictions = model([features, relations, index_x_valid, index_y_valid])

        valid_loss = loss(predictions, labels)
        return predictions, valid_loss