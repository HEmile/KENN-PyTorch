import torch
from kenn.KnowledgeEnhancer import KnowledgeEnhancer

class Kenn(torch.nn.Module):

    def __init__(self, predicates: [str],
                 clauses: [str],
                 activation=lambda x: x,
                 initial_clause_weight=0.5,
                 save_training_data=False):
        """Initialize the knowledge base.
        :param predicates: a list of predicates names
        :param clauses: a list of constraints. Each constraint is a string on the form:
        clause_weight:clause
        The clause_weight should be either a real number (in such a case this value is fixed) or an underscore
        (in this case the weight will be a tensorflow variable and learned during training).
        The clause must be represented as a list of literals separated by commas (that represent disjunctions).
        Negation must specified by adding the letter 'n' before the predicate name.
        An example:
           _:nDog,Animal
        """
        super().__init__()
        self.activation = activation
        self.knowledge_enhancer = KnowledgeEnhancer(
            predicates, clauses, initial_clause_weight, save_training_data)

    def forward(self, inputs: torch.Tensor) -> (torch.Tensor, [torch.Tensor, torch.Tensor]):
        """Improve the satisfaction level of a set of clauses.
        :param ground_atoms: the tensor containing the pre-activation values of the ground atoms
        :return: final preactivations"""
        deltas, deltas_list = self.knowledge_enhancer(inputs)
        return self.activation(inputs + deltas), deltas_list