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


from stanza import Pipeline
from ..utils.pos import penn2morphy, morphy2gwn, WNPOS
from ..database.db import RDFDatabase
from tqdm import tqdm

import pandas as pd
from io import StringIO

# Path: wordnet-gibberish/wordnet_gibberish/connections/connect_concepts.py

def find_related_concepts(definition: str, pipeline: Pipeline, rdfdb: RDFDatabase,
                          top_words : set[str] = set(["thing", "person", "place", "time", "way", "day"]),
                        ) -> list[dict]:
    """Find related concepts to a given definition.

    Args:
        definition (str): Definition of the concept.
        pipeline (stanza.Pipeline): Stanza pipeline for tokenization, MWT and POS and Lemma tagging.
        rdfdb (RDFDatabase): RDF database.
        top_words (set[str], optional): Set of top words to filter from. If a word from the definition is found in it, it will not be considered.
                                            Defaults to set(["thing", "person", "place", "time", "way", "day"]).
    Returns:
        list[dict]: List of related concepts. Each concept is a dictionary with the keys "token", "modify", "lemma", "pos" and "related" if "modify" field is True.
    """
    doc = pipeline(definition)
    parsed = [
        (word.text, word.lemma, word.xpos) for sent in doc.sentences for word in sent.words
    ]
    new_pos = []
    # Find the related concepts.
    for token, lemma, tag in parsed:

        # a/an
        if tag == "DT" and (token == "a" or token == "an"):
            new_pos.append({
                "token": token,
                "modify": False,
                "lemma": lemma,
                "pos": tag,
            })

        # Nouns or adjectives
        elif tag.startswith('NN') or tag.startswith('JJ'):
            # Is the lemma a commonly used word?
            if lemma in top_words:
                new_pos.append({
                    "token": token,
                    "modify": False,
                    "lemma": lemma,
                    "pos": tag,
                })
            else:
                # Find the definition of this word.
                morphy_tag = penn2morphy(tag)

                if morphy_tag != WNPOS.ADJ:
                    gwn_tag = morphy2gwn(morphy_tag)
                    answer = pd.read_csv(
                        StringIO(rdfdb.get_concept_and_def(lemma, gwn_tag)), sep="\t"
                    )
                    concepts = answer["?C"].to_list()
                else:
                    concepts = []
                    # Check the case if it's a comparative or a superlative.
                    # If that is the case, then we need to get the lemma as the stem
                    if tag == "JJR" and lemma.endswith("er"):
                        lemma = lemma[:-2]
                    elif tag == "JJS" and lemma.endswith("est"):
                        lemma = lemma[:-3]
                    for morphy_tag in [WNPOS.ADJ, WNPOS.ADJ_SAT]:
                        gwn_tag = morphy2gwn(morphy_tag)
                        answer = pd.read_csv(
                            StringIO(rdfdb.get_concept_and_def(lemma, gwn_tag)), sep="\t"
                        )
                        concepts.extend(answer["?C"].to_list())
                        
                # These things are the related concepts.
                new_pos.append({
                    "token": token,
                    "modify": True,
                    "lemma": lemma,
                    "pos": tag,
                    "related": concepts.copy()
                })
        
        elif tag.startswith('VB'):
            if lemma in top_words:
                new_pos.append({
                    "token": token,
                    "modify": False,
                    "lemma": lemma,
                    "pos": tag,
                })
            else:
                if tag == "VBN":
                    concepts = []
                    for morphy_tag in [WNPOS.VERB, WNPOS.ADJ, WNPOS.ADJ_SAT]:
                        gwn_tag = morphy2gwn(morphy_tag)
                        answer = pd.read_csv(
                            StringIO(rdfdb.get_concept_and_def(lemma, gwn_tag)), sep="\t"
                        )
                        concepts.extend(answer["?C"].to_list())
                
                else:
                    gwn_tag = morphy2gwn(WNPOS.VERB)
                    answer = pd.read_csv(
                        StringIO(rdfdb.get_concept_and_def(lemma, gwn_tag)), sep="\t"
                    )
                    concepts = answer["?C"].to_list()
                new_pos.append({
                    "token": token,
                    "modify": True,
                    "lemma": lemma,
                    "pos": tag,
                    "related": concepts.copy()
                })

    return new_pos

def create_connectivity_table(concepts_df):
    connectivity_table = []
    concepts = set(concepts_df.index)
    for idx, row in tqdm(concepts_df.iterrows(), total=len(concepts_df), desc="Creating connectivity table"):
        for entry in row["connected"]:
            # The "related" field of entry is a list of concepts

            if entry["modify"]:
                connectivity_table.append({
                    "concept": idx,
                    "definition": row["definition"],
                    "token": entry["token"],
                    "lemma": entry["lemma"],
                    "pos": entry["pos"],
                    "related": [
                        x for x in entry["related"] if x in concepts
                    ]
                })

    connectivity_df = pd.DataFrame(connectivity_table)
    return connectivity_df

def update_kg(connectivity_df, rdfdb, verbose=True):
    # Now, we can add the connections to the KG.
    new_triples = []
    for i, x in tqdm(connectivity_df.iterrows(),
                     total=len(connectivity_df),
                     desc="Creating new triples",
                     disable = not verbose):
        x_id = x["concept"].split("oewn-")[-1][:len("15022096")]
        lemma = x["lemma"]
        blank_node = f"""_:_{lemma.replace(' ', '').replace("'", "")}_{x_id}"""
        if len(x["related"]) == 0:
            continue
        new_triples.append(
            "\t".join([x["concept"], "<http://shortcut.org/terms/definitionWord>", blank_node, "."])
            )
        new_triples.append(
            "\t".join([blank_node, "<http://www.w3.org/1999/02/22-rdf-syntax-ns#value>", f'"{lemma}"@en', "."])
            )
        for r in x["related"]:
            new_triples.append(
                "\t".join([blank_node, "<http://shortcut.org/terms/references>", r, "."])
                )
            
    # Add the new triples to the KG
    new_triples = "\n".join(new_triples)
    rdfdb.add_data(new_triples)
        
