"""
Parsing with Context Free Grammars
Name: Brian Yang
"""

import sys
from collections import defaultdict
from math import fsum
import math


class Pcfg(object):
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file):
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None
        self.read_rules(grammar_file)

    def read_rules(self, grammar_file):

        for line in grammar_file:
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line:
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else:
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()

    def parse_rule(self, rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";", 1)
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # TODO, Part 1

        # CNF form 
        # rules are either
        # A-> B C      A, B, C are non-terminals (upper case)
        # or
        # A -> b       # A is non-terminal (upper case)
        # b is a terminal (lower case)

        # first check for valid form
        for lhs, list_of_rules in self.lhs_to_rules.items():

            total_prob = 0
            for rule in list_of_rules:

                left = rule[0]
                rhs = rule[1]
                probability = rule[2]

                # check for proper CNF form
                # first check if left is non-terminal
                if not (left.isupper()):
                    print("Grammar is not in CNF form. LHS is not a non-terminal.")
                    return False
                else:
                    # now check for valid rhs form
                    if len(rhs) == 2:
                        # 1. A -> B C
                        if (not rhs[0].isupper()) or (not rhs[1].isupper()):
                            print("Grammar is not in CNF form. RHS is not both non-terminal")
                            return False
                    elif len(rhs) == 1:
                        # 2. A -> b
                        if rhs[0].isupper():
                            print("Grammar not in CNF form. RHS is a single non-terminal")
                            return False

                # add probability to running sum for the unique lhs
                total_prob += probability

            if not math.isclose(total_prob, 1):
                print("Grammar not valid PCFG. Sum of probabilities for this lhs is not 1")
                return False

        return True


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as grammar_file:
        grammar = Pcfg(grammar_file)

        if grammar.verify_grammar():
            print("Valid grammar")
        else:
            print("Invalid grammar")

        print("START Symbol: ", grammar.startsymbol)
