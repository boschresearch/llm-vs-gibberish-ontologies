# Experiment resources related to the paper "Do LLMs Really Adapt to Domains? An Ontology Learning Perspective"
# (ISWC 2024)
# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from .utils import isVowel

def conjugate(gibberish_verb, tag):
    """Conjugates a verb according to its Penn Treebank POS tag.

    Args:
        gibberish_verb (str): Verb to conjugate.
        tag (str): Penn Treebank POS tag of the verb. It can be "VBD", "VBN", "VBG", "VBZ", "VB", "VBP", "VBG".

    Returns:
        str: Conjugated verb.
    """
    if tag == "VBD" or tag == "VBN":
        if gibberish_verb.endswith("ss") or gibberish_verb.endswith("sh") or gibberish_verb.endswith("ch") or gibberish_verb.endswith("x") \
            or gibberish_verb.endswith("o") or gibberish_verb.endswith("z"):
            new_word = gibberish_verb + "ed"
        elif gibberish_verb[-1] == "y" and not(isVowel(gibberish_verb[-2])):
            new_word = f'{gibberish_verb[:-1]}ied'
        else:
            new_word = gibberish_verb + ('ed' if gibberish_verb[-1] != 'e' else 'd')
        

    elif tag == "VBG":
        new_word = (gibberish_verb[:-1] if isVowel(gibberish_verb[-1]) else gibberish_verb) + "ing"
    elif tag == "VBZ":
        if gibberish_verb.endswith("ss") or gibberish_verb.endswith("sh") or gibberish_verb.endswith("ch") or gibberish_verb.endswith("x") \
            or gibberish_verb.endswith("o") or gibberish_verb.endswith("z"):
            new_word = gibberish_verb + "es"
        elif gibberish_verb[-1] == "y" and not(isVowel(gibberish_verb[-2])):
            new_word = f'{gibberish_verb[:-1]}ies'
        else:
            new_word = gibberish_verb + 's'
    else:
        new_word = gibberish_verb
    return new_word

def plural_accord(gibberish_noun, tag):
    """Pluralizes a noun according to its Penn Treebank POS tag.

    Args:
        gibberish_noun (str): Noun to pluralize.
        tag (str): Penn Treebank POS tag of the noun. It can be "NNS" or "NNPS".

    Returns:
        str: Pluralized noun.
    """
    if tag == "NNS" or tag == "NNPS":
        if gibberish_noun.endswith("ss") or gibberish_noun.endswith("sh") or gibberish_noun.endswith("x") or gibberish_noun.endswith("ch") :
            new_word = gibberish_noun + "es"
        elif gibberish_noun[-1] == "y" and not(isVowel(gibberish_noun[-2])):
            new_word = gibberish_noun + "es"
        elif gibberish_noun[-1] == "o" and not(isVowel(gibberish_noun[-2])):
            new_word = gibberish_noun + "es"
        elif gibberish_noun[-1] == "s" or gibberish_noun[-1] == "z":
            new_word = f"""{gibberish_noun}{gibberish_noun[-1]}es"""
        elif gibberish_noun.endswith("fe"):
            new_word = gibberish_noun[:-2] + "ves"
        elif gibberish_noun[-1] == "f" and gibberish_noun[-2] != "f":
            new_word = gibberish_noun[:-1] + "ves"
        else:
            new_word = gibberish_noun + "s"
        return new_word
    else:
        return gibberish_noun


def adjective_accord(gibberish_adj, tag):
    """Adjective according to its Penn Treebank POS tag.

    Args:
        gibberish_adj (str): Adjective to accord.
        tag (str): Penn Treebank POS tag of the adjective. It can be "JJR" or "JJS".
    
    Returns:
        str: Accorded adjective.
    """
    if tag == "JJR":
        # Comparative
        if gibberish_adj.endswith("e"):
            new_word = gibberish_adj + "r"
        elif gibberish_adj.endswith("y") and not(isVowel(gibberish_adj[-2])):
            new_word = f"""{gibberish_adj[:-1]}ier"""
        else:
            new_word = gibberish_adj + "er"
        return new_word
    elif tag == "JJS":
        # Superlative
        if gibberish_adj.endswith("e"):
            new_word = gibberish_adj + "st"
        elif gibberish_adj.endswith("y") and not(isVowel(gibberish_adj[-2])):
            new_word = f"""{gibberish_adj[:-1]}iest"""
        else:
            new_word = gibberish_adj + "est"
    else:
        return gibberish_adj