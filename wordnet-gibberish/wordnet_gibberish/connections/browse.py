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


from ..database.db import RDFDatabase
import pandas as pd
import queue
from tqdm import tqdm


class Pickaxe:
    def __init__(self, rdfdb: RDFDatabase,
                 root_concepts: list[str]):
        """Class to find related concepts to a given definition.

        Args:
            rdfdb (RDFDatabase): The RDF database object.
            root_concepts (list[str]): The root concept URIs to start from.
        """
        
        self.rdfdb = rdfdb
        self.root_lexical_concepts = root_concepts

    def get_horizontal_concepts(self, concept : str) -> pd.DataFrame:
        """Get the horizontal relations of a concept.

        Args:
            concept (str): The concept URI.
        
        Returns:
            pd.DataFrame: The DataFrame with the horizontal relations. Columns are ?C for the concept URI, ?P for the part of speech, ?D for the definition, and ?F for the forms.
        """

        query = f"""
        SELECT DISTINCT ?C ?P (STR(?Dp) AS ?D) (GROUP_CONCAT(DISTINCT ?G; SEPARATOR=";") AS ?F)
        WHERE {{
            ?C sct:derivation {concept} .
            ?C rdf:type ontolex:LexicalConcept .
            ?C gwn:partOfSpeech ?P .
            ?C sct:definition ?Dp .
            ?C sct:writtenForm ?G .
            FILTER(LANGMATCHES(LANG(?G), "en")  && LANGMATCHES(LANG(?Dp), "en"))
        }} GROUP BY ?C ?P ?Dp
        """
        return self.rdfdb.issue_query(query, to_table=True)

    def mine(self, max_depth: int = 5, domain_topic=False, meronyms=False,
             verbose : bool = False) -> dict:
        """Mine the WordNet for related (hyponymous or derivated) concepts of the registered root concepts.
        
        Args:
            max_depth (int, optional): The maximum depth to mine. Defaults to 5.
            domain_topic (bool, optional): Whether to consider domain topics. Defaults to False.
            meronyms (bool, optional): Whether to consider meronyms. Defaults to False.
            verbose (bool, optional): Whether to show a progress bar. Defaults to False.
        
        Returns:
            dict: A dictionary with the concept URIs as keys and the depth, part of speech, forms, and definition as values.
        """

        assert max_depth > 0, "max_depth must be greater than 0"

        # First step: top concepts

        explored_concepts = {
            c : {
                "depth" : 0
            }  for c in self.root_lexical_concepts
        }

        query = f"""
        SELECT DISTINCT ?O ?P (STR(?Dp) AS ?D) (GROUP_CONCAT(DISTINCT ?G; SEPARATOR=";") AS ?F)
        WHERE {{
            VALUES ?O {{ {" ".join(self.root_lexical_concepts)} }}
            ?O gwn:partOfSpeech ?P .
            ?O sct:definition ?Dp .
            ?O sct:writtenForm ?G .
            FILTER(LANGMATCHES(LANG(?G), "en")  && LANGMATCHES(LANG(?Dp), "en"))
        }} GROUP BY ?O ?P ?Dp
        """
        df = self.rdfdb.issue_query(query, to_table=True)
        for _, x in df.iterrows():
            explored_concepts[x["?O"]] = {
                "depth" : 0,
                "pos" : x["?P"],
                "forms" : x["?F"].split(";"),
                "definition" : x["?D"]
            }


        for root_lexical_concept in tqdm(self.root_lexical_concepts,
                                         total=len(self.root_lexical_concepts),
                                         disable = not verbose):
            q = queue.Queue()
            q.put((root_lexical_concept, 0))


            while not q.empty():
                current_concept_uri, current_depth = q.get()

                if current_depth >= max_depth:
                    continue

                # Get vertical relations

                query = f"""
                SELECT DISTINCT ?O ?P (STR(?Dp) AS ?D) (GROUP_CONCAT(DISTINCT ?G; SEPARATOR=";") AS ?F)
                WHERE {{
                    {current_concept_uri} gwn:hyponym{' | gwn:has_domain_topic' if domain_topic else ''}{' | gwn:meronym' if meronyms else ''} ?O .
                    ?O rdf:type ontolex:LexicalConcept .
                    ?O gwn:partOfSpeech ?P .
                    ?O sct:definition ?Dp .
                    ?O sct:writtenForm ?G .
                    FILTER(LANGMATCHES(LANG(?G), "en") && LANGMATCHES(LANG(?Dp), "en"))
                }} GROUP BY ?O ?P ?Dp
                """

                df = self.rdfdb.issue_query(query, to_table=True)
                df["?F"] = df["?F"].apply(lambda x: x.split(";"))

                # Add concepts to explored concepts
                for concept, pos, lemma, definition in zip(
                    df["?O"].values, df["?P"].values, df["?F"].values, df["?D"].values
                    ):
                    # Queue if not seen
                    pos = pos.split("https://globalwordnet.github.io/schemas/wn#")[-1][:-1]

                    if (not concept in explored_concepts):
                        q.put((concept, current_depth+1))
                        explored_concepts[concept] = {"depth" : current_depth + 1,
                                                      "pos" : pos,
                                                      "forms" : lemma,
                                                      "definition" : definition
                                                      }
                    # If seen, but at a higher depth before:
                    elif (concept in explored_concepts and (current_depth + 1) < explored_concepts[concept]["depth"]):
                        # Queue again
                        q.put((concept, current_depth+1))
                        explored_concepts[concept]["depth"] = current_depth + 1
                    # Otherwise, do nothing.

                    assert concept in explored_concepts, f"Concept {concept} not in explored concepts."    
                    
                # Get horizontal relations
                df = self.get_horizontal_concepts(current_concept_uri)
                df["?F"] = df["?F"].apply(lambda x: x.split(";"))

                # Add concepts to explored concepts
                for concept, pos, lemma, definition in zip(
                        df["?C"].values, df["?P"].values, df["?F"].values, df["?D"].values
                    ):
                    # Queue if not seen
                    pos = pos.split("https://globalwordnet.github.io/schemas/wn#")[-1][:-1]
                    if (not concept in explored_concepts):
                        q.put((concept, current_depth+1))
                        explored_concepts[concept] = {"depth" : current_depth + 1,
                                                      "pos" : pos,
                                                      "forms" : lemma,
                                                      "definition" : definition}
                    # If seen, but at a higher depth before:
                    elif (concept in explored_concepts and (current_depth + 1) < explored_concepts[concept]["depth"]):
                        # Queue again
                        q.put((concept, current_depth+1))
                        explored_concepts[concept]["depth"] = current_depth + 1
                    # Otherwise, do nothing.
        
        # explored_concepts is a dictionary with the concept URIs as keys and the depth as values.
        return explored_concepts
