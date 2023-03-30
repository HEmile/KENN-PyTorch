from kenn import Kenn, KnowledgeEnhancer, RelationalKenn

from kenn.boost_functions import GodelBoostConormApprox, GodelBoostConorm, GodelBoostResiduum

def unary_parser(knowledge_file: str, activation=lambda x: x, initial_clause_weight=0.5, save_training_data=False, boost_function=GodelBoostConormApprox):
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

    return Kenn(predicates, clauses, activation, initial_clause_weight, save_training_data, boost_function=boost_function)


def unary_parser_ke(knowledge_file: str, initial_clause_weight=0.5, boost_function=GodelBoostConormApprox):
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

    return KnowledgeEnhancer(predicates, clauses, initial_clause_weight, boost_function=boost_function)


def relational_parser(knowledge_file: str, activation=lambda x: x, initial_clause_weight=0.5, boost_function=GodelBoostConormApprox):
    """
    Takes in input the knowledge file containing both unary and binary clauses and returns a RelationalKenn
    Layer, with input the predicates and clauses found in the knowledge file.
    :param knowledge_file: path of the prior knowledge file;
    """
    with open(knowledge_file, 'r') as kb_file:
        unary_literals_string = kb_file.readline()
        binary_literals_string = kb_file.readline()

        kb_file.readline()
        clauses = kb_file.readlines()

    u_groundings = [u + '(x)' for u in unary_literals_string[:-1].split(',')]
    b_groundings = [u + '(x)' for u in unary_literals_string[:-1].split(',')] + \
                   [u + '(y)' for u in unary_literals_string[:-1].split(',')] + \
                   [b + '(x.y)' for b in binary_literals_string[:-1].split(',')] + \
                   [b + '(y.x)' for b in binary_literals_string[:-1].split(',')]

    unary_clauses = []
    binary_clauses = []
    implication_clauses = []

    reading_unary = True
    reading_binary = True
    for clause in clauses:
        if clause[0] == '>' and not reading_unary:
            reading_binary = False
            continue
        if clause[0] == '>':
            reading_unary = False
            continue

        if reading_unary:
            unary_clauses.append(clause)
        elif reading_binary:
            binary_clauses.append(clause)
        else:
            implication_clauses.append(clause)

    return RelationalKenn(
        u_groundings,
        b_groundings,
        unary_clauses,
        binary_clauses,
        implication_clauses,
        activation,
        initial_clause_weight,
        boost_function=boost_function)
