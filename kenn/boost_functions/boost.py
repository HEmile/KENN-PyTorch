import torch
import abc
from torch.nn.functional import softmax
import numpy as np

class BoostFunction(torch.nn.Module, abc.ABC):

    def __init__(self, initial_weight: float, fixed_weight: bool,
                 min_weight, max_weight):
        super().__init__()
        self.register_parameter(
            name='clause_weight',
            param=torch.nn.Parameter(torch.tensor(initial_weight)))

        self.clause_weight.requires_grad = not fixed_weight
        self.min_weight = min_weight
        self.max_weight = max_weight


    def forward(self, selected_predicates: torch.Tensor, signs: torch.Tensor):
        """
        :param selected_predicates: [b, l] The pre-activations of the selected ground atoms. Signs are not applied
        :param signs: [l] The signs of the literals
        :return: [b, l] The delta given by the boost function
        """
        pass


class GodelBoostConormApprox(BoostFunction):

    def forward(self, selected_predicates: torch.Tensor, signs: torch.Tensor):
        self.clause_weight.data = torch.clip(self.clause_weight, self.min_weight, self.max_weight)

        clause_matrix = selected_predicates * signs
        # Approximated Godel t-conorm boost function on preactivations
        return signs * softmax(clause_matrix, dim=-1) * self.clause_weight


class GodelBoostResiduum(BoostFunction):

    def forward(self, selected_predicates: torch.Tensor, signs: [torch.Tensor, torch.Tensor]):
        self.clause_weight.data = torch.clip(self.clause_weight, self.min_weight, self.max_weight)

        antecedent_matrix = selected_predicates[0] * signs[0]
        consequent_matrix = selected_predicates[1] * signs[1]

        conjunction_val = torch.min(antecedent_matrix, 1)[0]
        disjunction_val = torch.max(consequent_matrix, 1)[0]
        indices_consequent = torch.argmax(consequent_matrix, 1) + antecedent_matrix.size()[1]

        # Godel residuum boost function on preactivations
        formula_satisfied = disjunction_val < conjunction_val
        delta = torch.zeros(antecedent_matrix.size()[0], antecedent_matrix.size()[1] + consequent_matrix.size()[1])
        delta[np.arange(delta.size()[0]), indices_consequent] = torch.minimum(self.clause_weight, conjunction_val - disjunction_val) * formula_satisfied * signs[1]

        return delta


class GodelBoostConorm(BoostFunction):

    def forward(self, selected_predicates: torch.Tensor, signs: torch.Tensor):
        self.clause_weight.data = torch.clip(self.clause_weight, self.min_weight, self.max_weight)
        clause_matrix = selected_predicates * signs
        # clause_matrix = torch.sigmoid(selected_predicates * signs)
        indices = torch.argmax(clause_matrix, -1)
        delta = torch.zeros_like(clause_matrix)
        # delta[indices] = torch.minimum(self.clause_weight, 1 - clause_matrix[indices])
        delta[np.arange(delta.size()[0]), indices] = self.clause_weight * signs[indices]

        # Approximated Godel t-conorm boost function on preactivations
        return delta


class LukasiewiczBoostConorm(BoostFunction):

    def __init__(self, initial_weight: float, fixed_weight: bool, min_weight, max_weight):
        super().__init__(initial_weight, fixed_weight, 0.0, 1.0)

    def forward(self, selected_predicates: torch.Tensor, signs: torch.Tensor):
        self.clause_weight.data = torch.clip(self.clause_weight, self.min_weight, self.max_weight)

        clause_matrix = (signs < 0) + signs * selected_predicates
        sums = torch.sum(clause_matrix, 1)
        return torch.ones(clause_matrix.shape) * ((sums < 1) *
                                                  (1 - sums) *
                self.clause_weight / clause_matrix.shape[1] )[:, None] * signs


class ProductBoostConorm(BoostFunction):

    def forward(self, selected_predicates: torch.Tensor, signs: torch.Tensor):

        self.clause_weight.data = torch.clip(self.clause_weight, self.min_weight, self.max_weight)
        clause_matrix = selected_predicates * signs

        truth_max, i = clause_matrix.max(dim=1)  # m: maximum literals' truth values, i: the corresponding vector of indexes
        truth = torch.sigmoid(clause_matrix)  # Activations of the literals' truth values
        new_truth_max = torch.sigmoid(truth_max + self.clause_weight)  # Activations of the highest literals after adding the delta
        delta_max = new_truth_max - truth[np.arange(truth.shape[0]), i]  # Deltas on the activations of the highest literal
        c = delta_max * (1 - new_truth_max) # c = delta_i * (1 - truth_i - delta_i) for all i

        deltas = (1. + truth) / 2. - torch.sqrt((
                                        (1. - truth + 0.0001) / 2.) ** 2
                                        - c[:, None])

        preactivation_delta = torch.logit(deltas, eps=0.0001) - clause_matrix
        preactivation_delta[np.arange(truth.shape[0]), i] = self.clause_weight

        return preactivation_delta * signs