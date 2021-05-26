"""
Parsing with Probabilistic Context Free Grammars
Name: Brian Yang
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg


### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict):
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) == 2 and \
                isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write(
                    "Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str):  # Leaf nodes may be strings
                continue
            if not isinstance(bps, tuple):
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(
                        bps))
                return False
            if len(bps) != 2:
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(
                        bps))
                return False
            for bp in bps:
                if not isinstance(bp, tuple) or len(bp) != 3:
                    sys.stderr.write(
                        "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(
                            bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write(
                        "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(
                            bp))
                    return False
    return True


def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict):
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) == 2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write(
                    "Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True


class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar):
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self, tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2

        # implementing the CYK algorithm
        # parse table is (n+1) x (n+1)
        n = len(tokens)
        parse_table = dict()
        for i in range(n + 1):
            parse_table[i] = dict()
            for j in range(n + 1):
                parse_table[i][j] = set()

        # initialization
        for i in range(n):
            s_i = tokens[i]
            # get all rules of the form A -> s_i
            rules_list = self.grammar.rhs_to_rules[(s_i,)]
            # create set of all As such that A -> s_i
            # A = rule[0]
            lhs_list = set([rule[0] for rule in rules_list])
            parse_table[i][i + 1] = lhs_list

        # begin main loop
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length

                for k in range(i + 1, j):
                    M = set()
                    # find all rules A -> B C where
                    # B in parse_table[i][k] and
                    # C in parse_table[k][j]
                    for A, list_of_rules in self.grammar.lhs_to_rules.items():
                        for rule in list_of_rules:
                            rhs = rule[1]
                            if len(rhs) == 2:
                                # rule is of the form A -> B C
                                # check if B is in parse_table[i][k]
                                # and if C is in parse_table[k][j]
                                B = rhs[0]
                                C = rhs[1]
                                if B in parse_table[i][k] and C in parse_table[k][j]:
                                    M.add(A)

                    parse_table[i][j] = parse_table[i][j].union(M)
        # end main loop

        return True if (self.grammar.startsymbol in parse_table[0][n]) else False

    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3

        n = len(tokens)
        table = dict()
        probs = dict()

        # initialization
        for i in range(n):
            table[(i, i + 1)] = dict()
            probs[(i, i + 1)] = dict()

            s_i = tokens[i]

            # all rules in rules_list are of the form A -> s_i
            rules_list = self.grammar.rhs_to_rules[(s_i,)]

            for rule in rules_list:
                A = rule[0]
                log_prob = math.log2(rule[2])
                probs[(i, i + 1)][A] = log_prob
                table[(i, i + 1)][A] = s_i

        # begin main loop
        for length in range(2, n + 1):
            for i in range(n - length + 1):

                j = i + length

                table[(i, j)] = dict()
                probs[(i, j)] = dict()

                # iterate through all possible splits for X -> B C in spans (i,k), (k,j)
                for k in range(i + 1, j):
                    B_list = table[(i, k)]
                    C_list = table[(k, j)]

                    for B in B_list:
                        for C in C_list:
                            rhs = (B, C)

                            # get all rules of the form X -> B C
                            rules_list = self.grammar.rhs_to_rules[rhs]

                            # for each rule X -> B C, update table[(i,j)][X] and probs[(i,j)][X]
                            for rule in rules_list:
                                X = rule[0]
                                log_prob = math.log2(rule[2])

                                curr_log_prob = log_prob + probs[(i, k)][B] + probs[(k, j)][C]

                                if X not in probs[(i, j)]:
                                    probs[(i, j)][X] = curr_log_prob
                                    table[(i, j)][X] = ((B, i, k), (C, k, j))
                                else:
                                    if curr_log_prob > probs[(i, j)][X]:
                                        probs[(i, j)][X] = curr_log_prob
                                        table[(i, j)][X] = ((B, i, k), (C, k, j))

        # end main loop
        return table, probs


def get_tree(chart, i, j, nt):
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4

    # chart[(i,j)][nt] contains the backpointer to the table entries 
    # that were used to create the nt phrase over the span (i,j) 

    # base case: nt is terminal
    if not nt.isupper():
        return nt

    # recursive case: nt is non-terminal
    # backpointer format: (nt, i, j)
    children = chart[(i, j)][nt]
    if isinstance(children, str):
        # leaf node
        return nt, children

    # non-leaf node -> get left and right child
    left_child = children[0]
    left_nt = left_child[0]
    left_i = left_child[1]
    left_j = left_child[2]

    right_child = children[1]
    right_nt = right_child[0]
    right_i = right_child[1]
    right_j = right_child[2]

    left_parse_tree = get_tree(chart, left_i, left_j, left_nt)
    right_parse_tree = get_tree(chart, right_i, right_j, right_nt)
    return nt, left_parse_tree, right_parse_tree


if __name__ == "__main__":
    filename = r"C:\Users\Brian\Desktop\Columbia University\Summer A 2021\Natural Language Processing\HW 2 COMS " \
               r"4705\hw2\atis3.pcfg "
    # with open('atis3.pcfg', 'r') as grammar_file:
    with open(filename, 'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        parser = CkyParser(grammar)
        toks_valid = ['flights', 'from', 'miami', 'to', 'cleveland', '.']

        print("------------------- Test 1 -------------------")
        print("String: ", toks_valid)
        print("Expected: is_in_language = True")
        print("Actual: is_in_language = ", parser.is_in_language(toks_valid))
        toks_invalid = ['miami', 'flights', 'cleveland', 'from', 'to', '.']

        print("------------------- Test 2 -------------------")
        print("String: ", toks_invalid)
        print("Expected: is_in_language = False")
        print("Actual: is_in_language = ", parser.is_in_language(toks_invalid))

        table, probs = parser.parse_with_backpointers(toks_valid)

        assert check_table_format(table)
        assert check_probs_format(probs)
        print("Table in correct format: ", check_table_format(table))
        print("Probs in correct format: ", check_probs_format(probs))

        print("\n\n\n")

        print(toks_valid)
        tup_actual = get_tree(table, 0, len(toks_valid), grammar.startsymbol)
        print(tup_actual)
        tup_expected = ('TOP', ('NP', ('NP', 'flights'), ('NPBAR', ('PP', ('FROM', 'from'), ('NP', 'miami')), ('PP', ('TO', 'to'), ('NP', 'cleveland')))), ('PUN', '.'))

        print("Correct") if tup_actual == tup_expected else print("Incorrect)")
