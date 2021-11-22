import torch
from numpy.typing import ArrayLike
from torch import Tensor
from kenn import relational_parser
from torch.nn.functional import softmax
from torch.nn import Linear, Dropout

from kenn.boost_functions import GodelBoostConormApprox, LukasiewiczBoostConorm, ProductBoostConorm


class Standard(torch.nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.h1 = Linear(in_features, 50)
        self.d1 = Dropout()
        self.h2 = Linear(50, 50)
        self.d2 = Dropout()
        self.h3 = Linear(50, 50)
        self.d3 = Dropout()
        self.last_layer = Linear(50, 6)

    def preactivations(self, inputs: torch.Tensor):
        x = torch.relu(self.h1(inputs))
        x = self.d1(x)
        x = torch.relu(self.h2(x))
        x = self.d2(x)
        x = torch.relu(self.h3(x))
        x = self.d3(x)

        return self.last_layer(x)

    def forward(self, inputs: torch.Tensor):
        z = self.preactivations(inputs)

        return z, softmax(z)


class Kenn(Standard):
    """
    Relational KENN Model with 3 KENN layers.
    """

    def __init__(self, knowledge_file: str, input_features: int, boost_function=ProductBoostConorm):
        super().__init__(input_features)
        self.knowledge = knowledge_file
        # There used to be 3 layers here. We keep to 1 for now. (This is apparently called 'greedy'
        self.kenn_layer_1 = relational_parser(self.knowledge, boost_function=boost_function)
        self.kenn_layer_2 = relational_parser(self.knowledge, boost_function=boost_function)
        self.kenn_layer_3 = relational_parser(self.knowledge, boost_function=boost_function)

    def forward(self, inputs: [Tensor, ArrayLike, Tensor, Tensor], save_debug_data=False, use_preactivations=True):
        # TODO: What to do with the save_debug_data argument?
        features = inputs[0]
        relations = inputs[1]
        sx = inputs[2]
        sy = inputs[3]

        if use_preactivations:
            z = self.preactivations(features)
        else:
            z = softmax(self.preactivations(features))
        z, _ = self.kenn_layer_1(z, relations, sx, sy)
        z, _ = self.kenn_layer_2(z, relations, sx, sy)
        z, _ = self.kenn_layer_3(z, relations, sx, sy)

        if use_preactivations:
            return softmax(z, dim=-1)
        else:
            return z
