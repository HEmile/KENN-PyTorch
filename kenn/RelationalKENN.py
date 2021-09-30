import torch
from kenn import KnowledgeEnhancer

class GroupBy(torch.nn.Module):
    """GroupBy layer
    """

    def __init__(self, number_of_unary_predicates: int):
        """Initialize the GroupBy layer.
        """
        super().__init__()
        self.n_unary = number_of_unary_predicates

    def forward(self, unary: torch.Tensor, deltas: torch.Tensor, index1, index2) -> (torch.Tensor, torch.Tensor):
        """Split the deltas matrix in unary and binary deltas.
        :param unary: [n, u] the tensor with unary predicates pre-activations. This is only used to get the shape from
        :param deltas: [n, 2u+b] the tensor containing the delta values
        :param index1: [n, b] a vector containing the indices of the first object
        of the pair referred by binary and deltas tensors
        :param index2: [n, b] a vector containing the indices of the second object
        of the pair referred by binary and deltas tensors
        :returns Two tensors. The second returns the binary deltas
        """
        ux = deltas[:, :self.n_unary]
        uy = deltas[:, self.n_unary:2 * self.n_unary]
        b = deltas[:, 2 * self.n_unary:]
        shape = unary.size()

        return torch.scatter(ux, 0, index1, torch.zeros(shape)) + torch.scatter(uy, 0, index2, torch.zeros(shape)), b

class Join(torch.nn.Module):
    """Join layer
    """

    def forward(self, unary: torch.Tensor, binary: torch.Tensor, index1: torch.Tensor, index2: torch.Tensor):
        """Join the unary and binary tensors.
        :param unary: the tensor with unary predicates pre-activations
        :param binary: the tensor with binary predicates pre-activations
        :param index1: a vector containing the indices of the first object
        of the pair referred by binary tensor
        :param index1: a vector containing the indices of the second object
        of the pair referred by binary tensor
        """

        index1 = torch.squeeze(index1)
        index2 = torch.squeeze(index2)

        # For the case where index1 and index2 are scalars, tf.squeeze will make them 0 dimensional
        if index1.ndim == 0 and index2.ndim == 0:
            index1 = torch.unsqueeze(index1, 0)
            index2 = torch.unsqueeze(index2, 0)

        return torch.cat([torch.gather(unary, 0, index1), torch.gather(unary, 0, index2), binary], dim=1)

class RelationalKenn(torch.nn.Module):

    def __init__(self, unary_predicates: [str],
                 binary_predicates: [str],
                 unary_clauses: [str],
                 binary_clauses: [str],
                 activation=lambda x: x,
                 initial_clause_weight=0.5):
        """Initialize the knowledge base.
        :param unary_predicates: the list of unary predicates names
        :param binary_predicates: the list of binary predicates names
        :param unary_clauses: a list of unary clauses. Each clause is a string on the form:
        clause_weight:clause
        The clause_weight should be either a real number (in such a case this value is fixed) or an underscore
        (in this case the weight will be a tensorflow variable and learned during training).
        The clause must be represented as a list of literals separated by commas (that represent disjunctions).
        Negation must specified by adding the letter 'n' before the predicate name.
        An example:
           _:nDog,Animal
        :param binary_clauses: a list of binary clauses
        :param activation: activation function
        :param initial_clause_weight: initial value for the cluase weight (if clause is not hard)
        """

        super().__init__()

        self.unary_clauses = unary_clauses
        self.binary_clauses = binary_clauses
        self.activation = activation

        self.unary_ke = None
        self.binary_ke = None
        self.join = None
        self.group_by = None

        if len(self.unary_clauses) != 0:
            self.unary_ke = KnowledgeEnhancer(
                unary_predicates, self.unary_clauses, initial_clause_weight=initial_clause_weight)
        if len(self.binary_clauses) != 0:
            self.binary_ke = KnowledgeEnhancer(
                binary_predicates, self.binary_clauses, initial_clause_weight=initial_clause_weight)

            self.join = Join()
            self.group_by = GroupBy(len(unary_predicates))


    def forward(self, unary: torch.Tensor, binary: torch.Tensor, index1: torch.Tensor, index2: torch.Tensor) \
            -> (torch.Tensor, torch.Tensor):
        """Forward step of Kenn model for relational data.
        :param unary: the tensor with unary predicates pre-activations
        :param binary: the tensor with binary predicates pre-activations
        :param index1: a vector containing the indices of the first object
        of the pair referred by binary tensor
        :param index2: a vector containing the indices of the second object
        of the pair referred by binary tensor
        """

        if len(self.unary_clauses) != 0:
            deltas_sum, deltas_u_list = self.unary_ke(unary)
            u = unary + deltas_sum
        else:
            u = unary

        if len(self.binary_clauses) != 0 and len(binary) != 0:
            joined_matrix = self.join(u, binary, index1, index2)
            deltas_sum, deltas_b_list = self.binary_ke(joined_matrix)

            delta_up, delta_bp = self.group_by(u, deltas_sum, index1, index2)
        else:
            delta_up = torch.zeros(u.shape)
            delta_bp = torch.zeros(binary.shape)

        return self.activation(u + delta_up), self.activation(binary + delta_bp)

    # This doens't seem to have a PyTorch correspondence
    # def get_config(self):
    #     config = super(RelationalKenn, self).get_config()
    #     config.update({'unary_predicates': self.unary_predicates})
    #     config.update({'unary_clauses': self.unary_clauses})
    #     config.update({'binary_predicates': self.binary_predicates})
    #     config.update({'binary_clauses': self.binary_clauses})
    #     config.update({'activation': self.activation})
    #     config.update({'initial_clause_weight': self.initial_clause_weight})
    #
    #     return config