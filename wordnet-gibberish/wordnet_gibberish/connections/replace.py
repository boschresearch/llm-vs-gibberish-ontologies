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


from ..translator.translator import Translator
from ..database import RDFDatabase
from ..utils.utils import safety_check, shorten_uri, isVowel, escape_unicode
from ..utils.grammar import conjugate, plural_accord, adjective_accord
from typing import Optional
import pandas as pd
from io import StringIO
from tqdm import tqdm
from stanza import Pipeline
from time import sleep


class Replacer:
    def __init__(
            self, translator: Translator, rdfdb : RDFDatabase,
            concepts_df: pd.DataFrame,
            translator_kwargs: Optional[dict] = None
            ):
        self.translator = translator
        self.rdfdb = rdfdb
        self.concepts_df = concepts_df
        if translator_kwargs is not None:
            self.translator_kwargs = translator_kwargs
        else:
            self.translator_kwargs = {
                "do_sample": False,
                "max_tries": 5
            }


        new_rules = """
        # One-way reference for directed acyclic graphs.
        [?c, sct:references, ?d] :- [?c, rdf:type, ontolex:LexicalConcept], [?c, sct:definitionWord, ?x], [?x, sct:references, ?d], [?x, rdf:value, ?w], [?d, sct:definition, ?y], NOT FILTER(CONTAINS(?y, ?w)) .
        [?c, sct:directReferences, ?d] :- [?c, rdf:type, ontolex:LexicalConcept], [?c, sct:references, ?d], NOT [?d, sct:references, ?c].
        [?x, sct:directReferences, ?d] :- [?c, sct:directReferences, ?d], [?c, sct:definitionWord, ?x], [?x, sct:references, ?d].
        """
        rdfdb.add_data(new_rules)
    
    def replace_definition(self, concept: str, new_definition: str) -> None:
        """Replace the definition of a concept.
        
        Args:
            concept (str): Concept URI.
            new_definition (str): New definition.
        """
        triples = []
        concept_id = concept.split("oewn-")[-1][:len("15022096")]
        blank_node = f"_:def_{concept_id}_gibberish"
        triples.append(
            " ".join([concept, "<https://globalwordnet.github.io/schemas/wn#definition>", blank_node, "."])
        )
        triples.append(
            " ".join([blank_node, "<http://www.w3.org/1999/02/22-rdf-syntax-ns#value>", f'"{escape_unicode(new_definition)}"@fr', "."])
        )

        triples =  "\n".join(triples)

        # Adding triples to data store
        self.rdfdb.add_data(triples)
    
    def blacklist_fn(self, word: str) -> bool:
        """Check if a word is blacklisted(i.e. already exists in the KG)
        
        Args:
            word (str): Word to check.
        
        Returns:
            bool: True if the word exists, False otherwise.
        """
        # If the input word contains multiple words, we need to check that at least one does not exist.
        if len(word) == 0:
            return True
        
        for word_ in word.split(" "):
            answer = self.rdfdb.issue_query(
                f"""
                SELECT ?C WHERE {{
                    ?C sct:writtenForm "{escape_unicode(word_)}"@en .
                }}
                """
            )
            # If the word is in the KG, it is blacklisted
            answer = answer[3:-1]
        
            # If one of the words does not exist, return False immediately.
            if len(answer) == 0:
                return False
        # if all the words exist:
        return True
    
    def mark_as_labeled(self, concepts: set[str]) -> None:
        """Mark a set of concepts as labeled.
        
        Args:
            concepts (set[str]): Set of concepts to mark as labeled.
        """
        for concept in concepts:
            self.rdfdb.add_data(
                f"""
                {concept} rdf:type sct:LabeledConcept .
                """
            )

    def first_step(self):
        # First step: Check which words are at the top of the dependency tree.
        # They do not have any potential substitution.
        print("[INFO] Checking for words at the top of the dependency tree.")

        query="""
        SELECT DISTINCT ?C (GROUP_CONCAT(DISTINCT ?L;separator="|") AS ?F) (STR(?G) AS ?D) ?P WHERE {
            ?C sct:definition ?G.
            ?C sct:writtenForm ?L .
            ?C gwn:partOfSpeech ?P .
            MINUS {?C sct:definitionWord ?W}.
            FILTER(LANGMATCHES(LANG(?L), "en") && LANGMATCHES(LANG(?G), "en"))
        } GROUP BY ?C ?G ?P
        """
        explored_concepts = set(self.concepts_df.index)
        answer_df = self.rdfdb.issue_query(query, to_table=True)
        print("Before filter: ", answer_df.head())
        answer_df = answer_df[answer_df["?C"].apply(lambda x: x in explored_concepts)]
        print("After filter: ", answer_df.head())

        processed_concepts = set()
        all_concepts = set(self.concepts_df.index)

        for idx, row in tqdm(answer_df.iterrows(), total=len(answer_df), desc="First step"):
            # Check for existing gibberish forms
            gibberish_forms = self.rdfdb.check_for_gibberish(row["?C"])

            # If there are already gibberish forms, we skip this concept
            if len(gibberish_forms) == 0:
                
                # If there are no gibberish forms, we create them
                new_word = None
                made_new_word = False
                pos = row["?P"].split("wn#")[-1][:-1]
                # While we still have not made a new word
                patience = 5
                while not made_new_word and patience > 0:
                    # We translate the word based on its definition
                    # We will noise the word a bit if the previous attempt failed.
                    # TODO: implement self.translator_kwargs
                    new_word = self.translator.translate(
                        word = row["?F"].split("|")[0] + "toto" * (5 - patience),
                        definition = row["?D"],
                        pos = pos,
                        do_sample=False,
                        blacklist_fn=self.blacklist_fn,
                        max_tries=5
                    )

                    made_new_word = not new_word is None
                    if not made_new_word:
                        sleep(5)
                    patience -= 1
                if not made_new_word:
                    raise Exception(
                        f"Could not make a new word for {row['?F'].split('|')[0]} (concept {row['?C']})"
                    )
                
                # We add the new word to the KG
                self.rdfdb.add_gibberish_repr(new_word, [row["?C"]])
                # We replace the definition of the concept
            else:
                new_word = gibberish_forms[0]
            # If there is not already a gibberish definition, we must add one.
            if len(self.rdfdb.check_gibberish_def(row["?C"])) == 0:
                self.replace_definition(row["?C"], row["?D"])
            processed_concepts.add(row["?C"])
            # Mark as labeled
            self.mark_as_labeled([row["?C"]])
            # We also need to do something about the homonymous words
            homonyms = self.rdfdb.homonym_concepts(row["?C"], lang="en")
            homonyms = pd.read_csv(StringIO(homonyms), sep="\t")
            for homonym, definition in homonyms[["?C", "?D"]].values:
                homonym_gibberish_forms = self.rdfdb.check_for_gibberish(homonym)
                if len(homonym_gibberish_forms) == 0:
                    self.rdfdb.add_gibberish_repr(new_word, [homonym])
                    if not homonym in all_concepts:
                        self.replace_definition(homonym, definition)
                        processed_concepts.add(homonym)
                        self.mark_as_labeled([homonym])

        print(f"[INFO] First step completed. {len(processed_concepts)} concepts processed.")
        return processed_concepts
    
    def process_definition(self, concept: str, 
                        connectivity_table: pd.DataFrame,
                        nlp: Pipeline) -> str:
        """Process the definition of a concept.
        
        Args:
            concept (str): Concept URI.
            definition (str): Definition of the concept.
        
        Returns:
            str: Processed definition.
        """
        # Get the rows of the concept in reverse order of their appearance
        rows = connectivity_table[connectivity_table["concept"] == concept]
        n = rows.shape[0]
        definition = rows["definition"].values[0]

        # Process the definition in reverse order
        doc = nlp(definition)
        parsed = [
            (word.text, word.lemma, word.xpos) for sent in doc.sentences for word in sent.words
        ][::-1]

        # New sentence
        new_sentence = []
        i = n-1
        next_item = rows.iloc[i]
        for text, lemma, tag in parsed:
            if text == next_item["token"] and tag == next_item["pos"]:
                # We must turn it into gibberish
                # Get the corresponding related concepts
                related_concepts = next_item["related"]
                # Get the gibberish form of the words
                if len(related_concepts) == 0:
                    new_sentence.append(
                        next_item["token"]
                    )
                else:
                    # Get the first gibberish representation that we can
                    gibberish_repr = []
                    j = 0
                    while len(gibberish_repr) == 0 and j <= len(related_concepts) - 1:
                        related_concept = related_concepts[j]
                        # Be carefull: some definitions contain the word itself.
                        gibberish_repr = self.rdfdb.check_for_gibberish(related_concept)
                        j += 1

                    assert len(gibberish_repr) > 0, f"No gibberish representation for {related_concepts} with concept {concept} and definition {definition}."

                    # We must accord it using the POS
                    if tag.startswith("NN"):
                        gibberish_repr_ = plural_accord(gibberish_repr[0], tag)
                    elif tag.startswith("JJ"):
                        gibberish_repr_ = adjective_accord(gibberish_repr[0], tag)
                    elif tag.startswith("VB"):
                        gibberish_repr_ = conjugate(gibberish_repr[0], tag)
                    else:
                        gibberish_repr_ = gibberish_repr[0]

                    new_sentence.append(
                        gibberish_repr_
                    )
                i -= 1
                next_item = rows.iloc[i] if i >= 0 else {
                    "token": None,
                    "pos": None,
                }
            elif (text == "a" or text == "an") and tag == "DT":
                # We must accord it using the previous word
                new_sentence.append(
                    "an" if isVowel(new_sentence[-1][0]) else "a"
                )
            else:
                new_sentence.append(text)
        return " ".join(new_sentence[::-1])
    
    def next_step(self, processed_concepts : set[str],
                  connectivity_table : pd.DataFrame,
                  pipeline: Pipeline
                  ) -> set[str]:
        # Check for words which have dependencies ONLY with previously processed words
        print("[INFO] Checking for words with dependencies only with previously processed words.")
        # TODO: Make a better query in case this one is too long.
        # We will relax the assumption of labeled concept:
        # We will simply take words that have a gibberish representation into account.
        # If A refers to B, and B has a gibberish representation, then A should be processed
        # regardless of whether B has a gibberish definition or not.
        # We also relax the constraint of directed reference.
        query = """
        SELECT DISTINCT ?C (GROUP_CONCAT(DISTINCT ?L;separator="|") AS ?F) ?P WHERE {
            ?C rdf:type ontolex:LexicalConcept ;
                sct:writtenForm ?L ;
                gwn:partOfSpeech ?P ;

            FILTER NOT EXISTS {
                ?C rdf:type sct:LabeledConcept .
            }

            ?C sct:definitionWord ?W.
            ?W sct:references ?E .
            ?E sct:writtenForm ?Fe .
            
            # C should not have words V that do not have a French representation
            MINUS {
                ?C sct:definitionWord ?V .
                ?V sct:references ?A .
                FILTER NOT EXISTS {
                    ?A sct:writtenForm ?Fa .
                    FILTER(LANGMATCHES(LANG(?Fa), "fr"))
                }
                FILTER((?A != ?C))
            }
            FILTER ( LANGMATCHES(LANG(?Fe), "fr") )
        } GROUP BY ?C ?P
        """
        answer_df = self.rdfdb.issue_query(query, to_table=True)
        # Restrict to the concepts that have not been processed yet
        all_concepts = set(self.concepts_df.index)
        answer_df = answer_df[answer_df["?C"].apply(lambda x: x in all_concepts)]
        new_processed_concepts = set()

        for idx, row in tqdm(answer_df.iterrows(), total=len(answer_df), desc="Next step"):
            # Check for existing gibberish forms
            gibberish_forms = self.rdfdb.check_for_gibberish(row["?C"])
            gibberish_defs = self.rdfdb.check_gibberish_def(row["?C"])
            # If there are already gibberish forms and defs, we skip this concept
            if len(gibberish_forms) == 0 and len(gibberish_defs) == 0:

                new_word = None
                made_new_word = False
                pos = row["?P"].split("wn#")[-1][:-1]
                patience = 5
                while not made_new_word and patience > 0:
                    # We must rewrite the definition of the concept.
                    definition = connectivity_table[connectivity_table["concept"] == row["?C"]]["definition"].values[0]
                    # We translate the word based on its definition
                    # We will noise the word a bit if the previous attempt failed.
                    new_word = self.translator.translate(
                        word = row["?F"].split("|")[0] + "toto" * (5 - patience),
                        definition = definition,
                        pos = pos,
                        do_sample=False,
                        blacklist_fn=self.blacklist_fn,
                        max_tries=5
                    )
                    made_new_word = not new_word is None
                    patience -= 1
                if not made_new_word:
                    raise Exception(
                        f"Could not make a new word for {row['?F'].split('|')[0]} (concept {row['?C']})"
                    )
                
                # We add the new word to the KG
                self.rdfdb.add_gibberish_repr(new_word, [row["?C"]])
                # Note: if the definition of the word contains itself
                # (e.g. Vitamin B :  originally thought to be a single vitamin but now separated into several B vitamins)
                # We can get away by processing the representation first.
                new_definition = self.process_definition(
                    row["?C"], connectivity_table, pipeline
                )
                # We replace the definition of the concept
                self.replace_definition(row["?C"], new_definition)
            elif len(gibberish_forms) > 0:
                new_word = gibberish_forms[0]
                new_definition = self.process_definition(
                    row["?C"], connectivity_table, pipeline
                )
                # We replace the definition of the concept
                self.replace_definition(row["?C"], new_definition)
                
            new_processed_concepts.add(row["?C"])
            self.mark_as_labeled([row["?C"]])

            # We also need to do something about the homonymous words
            homonyms = self.rdfdb.homonym_concepts(row["?C"], lang="en")
            homonyms = pd.read_csv(StringIO(homonyms), sep="\t")
            if homonyms.shape[0] > 0:
                for homonym in homonyms["?C"].values:
                    homonym_gibberish_forms = self.rdfdb.check_for_gibberish(homonym)
                    if len(homonym_gibberish_forms) == 0:
                        self.rdfdb.add_gibberish_repr(new_word, [homonym])
                    new_processed_concepts.add(homonym)

        return new_processed_concepts
    
    def name_random(self, remaining_concepts : set[str]):
        """If there are still concepts to process but none can be resolved, we will sample one randomly and assign a gibberish representation without labeling it.

        Args:
            remaining_concepts (set[str]): Set of remaining concepts.
        """
        # Choose one concept randomly.
        # Do not turn into a set: this is a highly inefficient (linear time) way of doing it.
        # Instead, just pop an element and put it back.
        assert len(remaining_concepts) > 0
        concept = remaining_concepts.pop()
        remaining_concepts.add(concept)

        # Assign a gibberish name to it.
        gibberish_forms = self.rdfdb.check_for_gibberish(concept)
        word = self.concepts_df.loc[concept, "forms"][0]
        if len(gibberish_forms) == 0:
            # We must assign a gibberish representation to it.
            new_word = None
            made_new_word = False
            pos = self.concepts_df.loc[concept, "pos"]
            patience = 5
            while not made_new_word and patience > 0:
                # We must rewrite the definition of the concept.
                definition = self.concepts_df.loc[concept, "definition"]
                # We translate the word based on its definition
                # We will noise the word a bit if the previous attempt failed.
                new_word = self.translator.translate(
                    word = word + "toto" * (5 - patience),
                    definition = definition,
                    pos = pos,
                    do_sample=False,
                    blacklist_fn=self.blacklist_fn,
                    max_tries=5
                )
                made_new_word = not new_word is None
                patience -= 1

            if not made_new_word:
                raise Exception(
                    f"Could not make a new word for {word} (concept {concept})"
                )
        else:
            new_word = gibberish_forms[0]
        # We add the new word to the KG
        self.rdfdb.add_gibberish_repr(new_word, [concept])
        # But we don't do anything else.
        return concept



