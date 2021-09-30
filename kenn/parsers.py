from kenn.KnowledgeEnhancer import KnowledgeEnhancer
from kenn.KENN import Kenn


def unary_parser(knowledge_file: str, activation=lambda x: x, initial_clause_weight=0.5, save_training_data=False):
    """
    Takes in input the knowledge file containing only unary clauses and returns a Kenn Layer,
    with input the predicates and clauses found in the knowledge file.
    :param knowledge_file: path of the prior knowledge file
    """
    with open(knowledge_file, 'r') as kb_file:
        predicates_string = kb_file.readline()
        kb_file.readline()
        clauses = kb_file.readlines()

    predicates = predicates_string[:-1].split(',')

    return Kenn(predicates, clauses, activation, initial_clause_weight, save_training_data)


def unary_parser_ke(knowledge_file: str, initial_clause_weight=0.5):
    """
    Takes in input the knowledge file containing only unary clauses and returns a Knowledge Enhancer layer,
    with input the predicates and clauses found in the knowledge file.
    :param knowledge_file: path of the prior knowledge file;
    """
    with open(knowledge_file, 'r') as kb_file:
        predicates_string = kb_file.readline()
        kb_file.readline()
        clauses = kb_file.readlines()

    predicates = predicates_string[:-1].split(',')

    return KnowledgeEnhancer(predicates, clauses, initial_clause_weight)