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

