import torch
import numpy as np
from torch.nn.functional import softmax
from typing import Union

# TODO: No parallelization over clauses
class ClauseEnhancer(torch.nn.Module):

    def __init__(self,
                 available_predicates: [str],
                 clause_string: str,
                 initial_clause_weight: float,
                 min_weight=0,
                 max_weight=500):
        """Initialize the clause.
        :param available_predicates: the list of all possible literals in a clause
        :param clause_string: a string representing a conjunction of literals. The format should be:
        clause_weight:clause
        The clause_weight should be either a real number (in such a case this sign is fixed) or an underscore
        (in this case the weight will be learned during training).
        The clause must be represented as a list of literals separated by commas (that represent disjunctions).
        Negation must specified by adding the letter 'n' before the predicate name.
        An example:
           _:nDog,Animal
        :param initial_clause_weight: the initial value of the clause weight. Used if the clause weight is learned.
        """
        super().__init__()
        # Split weight and clause
        weight_clause_split = clause_string.split(':')

        self.clause_string = weight_clause_split[1]
        weight_string = weight_clause_split[0]
        self.string = weight_clause_split[1].replace(
            ',', 'v').replace('(', '').replace(')', '')

        if weight_string == '_':
            self.initial_weight = initial_clause_weight
            self.fixed_weight = False
        else:
            self.initial_weight = float(weight_string)
            self.fixed_weight = True

        literals = self.clause_string.split(',')
        self.number_of_literals = len(literals)

        # Setup indexing of the literals
        gather_literal_indices = []
        self.scatter_literal_indices = []
        signs = []

        for literal in literals:
            sign = 1
            if literal[0] == 'n':
                sign = -1
                literal = literal[1:]

            # This could be improved using a hashmap instead of a list
            literal_index = available_predicates.index(literal)
            gather_literal_indices.append(literal_index)
            # What's the difference? This just creates singletons of the same list as above.
            # gather is [n], scatter is [n, 1]...
            self.scatter_literal_indices.append([literal_index])
            signs.append(sign)

        self.gather_literal_indices = torch.tensor(gather_literal_indices)

        self.signs = torch.tensor(signs, dtype=torch.float32)

        self.register_parameter(
            name='clause_weight',
            param=torch.nn.Parameter(torch.tensor(self.initial_weight)))

        self.clause_weight.requires_grad = not self.fixed_weight
        self.min_weight = min_weight
        self.max_weight = max_weight

    def grounded_clause(self, ground_atoms: torch.Tensor) -> torch.Tensor:
        """Find the grounding of the clause
        :param ground_atoms: [b, 2|U| + |B|] the tensor containing the pre activations of the ground atoms
        :return: the grounded clause (a tensor with literals truth values)
        """

        # Choose the right literals
        # self.gather_literal_indices: [l], each in {1, ..., 2|U| + |B|}
        selected_predicates = ground_atoms[..., self.gather_literal_indices] # [b, l]
        clause_matrix = selected_predicates * self.signs

        return clause_matrix

    def forward(self, ground_atoms: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Improve the satisfaction level of the clause.
        :param ground_atoms: the tensor containing the pre-activation values of the ground atoms
        :return: delta vector to be summed to the original pre-activation tensor to obtain an higher satisfaction of \
        the clause"""
        # [b, l]
        clause_matrix = self.grounded_clause(ground_atoms)

        # Approximated Godel t-conorm boost function on preactivations
        # VERY BIG WARNING TODO WHY DOES IT MULTIPLY WITH self.signs AGAIN?? IT ALSO DOES IN GROUNDED_CLAUSE!!

        delta = self.signs * softmax(clause_matrix, dim=-1) * self.clause_weight

        # [b, 2|U|+|B|]
        scattered_delta = torch.zeros_like(ground_atoms)
        scattered_delta[..., self.gather_literal_indices] = delta

        return scattered_delta, delta

