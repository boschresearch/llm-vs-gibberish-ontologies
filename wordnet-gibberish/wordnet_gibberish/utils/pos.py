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


class WNPOS:
    NOUN = 'n'
    VERB = 'v'
    ADJ = 'a'
    ADV = 'r'
    ADJ_SAT = 's'

def morphy2penn(morphytag, returnNone=False):
    """Converts a WordNet POS tag to a Penn Treebank POS tag.

    Args:
        morphytag (str): WordNet POS tag.
        returnNone (bool, optional): Whether to return None if the tag is not found. Defaults to False.

    Returns:
        str or None: Penn Treebank POS tag. If the tag is not found, it returns None if returnNone is True, otherwise it returns an empty string.
    """
    penntag = {
        WNPOS.NOUN: "NN", WNPOS.ADJ : "JJ", WNPOS.ADJ_SAT : "JJ",
        WNPOS.VERB : "VB", WNPOS.ADV: "RB"
    }
    try:
        return penntag[morphytag]
    except:
        return None if returnNone else ''

def penn2morphy(penntag, returnNone=False):
    """Converts a Penn Treebank POS tag to a WordNet POS tag.

    Args:
        penntag (str): Penn Treebank POS tag.
        returnNone (bool, optional): Whether to return None if the tag is not found. Defaults to False.
    
    Returns:
        str or None: WordNet POS tag. If the tag is not found, it returns None if returnNone is True, otherwise it returns an empty string.
    """

    morphy_tag = {'NN':WNPOS.NOUN, 'JJ':WNPOS.ADJ,
                  'VB':WNPOS.VERB, 'RB':WNPOS.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ''

    
def morphy2gwn(morphytag, returnNone=False):
    """Converts a WordNet POS tag to a GWN POS tag.

    Args:
        morphytag (str): WordNet POS tag.
        returnNone (bool, optional): Whether to return None if the tag is not found. Defaults to False.

    Returns:
        str or None: GWN POS tag. If the tag is not found, it returns None if returnNone is True, otherwise it returns an empty string.
    """
    gwntag = {
        WNPOS.NOUN: "noun", WNPOS.ADJ : "adjective", WNPOS.ADJ_SAT : "adjective_satellite",
        WNPOS.VERB : "verb", WNPOS.ADV: "adverb"
    }
    try:
        return gwntag[morphytag]
    except:
        return None if returnNone else ''