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


import requests
import pandas as pd
from io import StringIO
from pathlib import Path
from typing import Optional, Union
import unidecode
from ..utils.utils import escape_unicode

WORDNET_PREFIXES = """PREFIX dct: <http://purl.org/dc/terms/>
PREFIX gwn: <https://globalwordnet.github.io/schemas/wn#>
PREFIX id: <https://en-word.net/id/>
PREFIX lemma: <https://en-word.net/lemma/>
PREFIX lemon: <http://www.w3.org/ns/lemon/>
PREFIX lime: <http://www.w3.org/ns/lemon/lime#>
PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfox: <https://rdfox.com/vocabulary#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX sh: <http://www.w3.org/ns/shacl#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX swrl: <http://www.w3.org/2003/11/swrl#>
PREFIX swrlb: <http://www.w3.org/2003/11/swrlb#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX sct: <http://shortcut.org/terms/>
"""

def assert_response_ok(response, message):
    if not response.ok:
        raise Exception(
            message + "\nStatus received={}\n{}".format(response.status_code, response.text)
            )

PATH_KG = Path(__file__).parent.parent.parent / "data" / "real" / "english-wordnet-2023.ttl"
PATH_RULES = Path(__file__).parent.parent.parent / "data" / "real" / "rules.dlog"

class RDFDatabase:
    def __init__(self, endpoint: str = "http://localhost:12110",
                 kg_name: str = "WordNet-english",
                 timeout: int = 30,
                 engine: str = "rdfox",
                 init_kg: bool = True,
                 add_prefixes: bool = True
                 ):
        """Initialize the RDF database query interface.

        Args:
            endpoint (str, optional): The endpoint of the RDF database. Defaults to "http://localhost:12110".
            kg_name (str, optional): The name of the knowledge graph. Defaults to "WordNet-english".
            timeout (int, optional): Timeout for the requests. Defaults to 30.
            engine (str, optional): The engine of the RDF database. Defaults to "rdfox".
            init_kg (bool, optional): Whether to initialize the knowledge graph. Defaults to True.
            add_prefixes (bool, optional): Whether to add the prefixes to the RDF database. Defaults to True.

        Raises:
            NotImplementedError: Only RDFox engine is supported at the moment.
        """
        self.endpoint = endpoint
        self.prefixes = WORDNET_PREFIXES
        self.kg_name = kg_name
        self.timeout = timeout
        self.engine = engine
        if not self.engine == "rdfox":
            raise NotImplementedError("Only RDFox engine is supported at the moment.")
        
        if init_kg:
            assert PATH_KG.exists(), f"The RDF database file does not exist at the path {PATH_KG}"
            assert PATH_RULES.exists(), f"The rules file does not exist at the path {PATH_RULES}"
            # Load the RDF database with the WordNet data
            print("[INFO] Loading the RDF database with the WordNet data...")
            with open(PATH_KG, "r") as f:
                self.add_data(f.read())
        # Load the RDF database with the rules
        print("[INFO] Loading the RDF database with the rules...")
        with open(PATH_RULES, "r") as f:
            self.add_data(f.read())

        # Add the prefixes to the RDF database
        if add_prefixes:
            prefixes_data = WORDNET_PREFIXES.replace("PREFIX", "@prefix").replace("\n", " .\n")
            self.add_data(prefixes_data)

        self.banned_chars = []

    def issue_query(self, query : str, to_table : bool = False, verbose : bool =False
                    ) -> Union[str, pd.DataFrame]:
        """Issue a SPARQL query to the RDF database.

        Args:
            query (str): SPARQL query to issue.
            to_table (bool, optional): Whether to return the result as a pandas DataFrame. Defaults to False.
            verbose (bool, optional): Whether to print the response. Defaults to False.

        Raises:
            NotImplementedError: Only RDFox engine is supported at the moment.

        Returns:
            str or pd.DataFrame: Response to the SPARQL query. If to_table is True, it returns a pandas DataFrame.
        """
        if self.engine == "rdfox":
            sparql_text = self.prefixes + "\nSELECT ?S ?P ?O WHERE { ?S ?P ?O . } LIMIT 1000" if query is None else query
            response = requests.get(
                f"{self.endpoint}/datastores/{self.kg_name}/sparql",
                params={"query": sparql_text},
                timeout=self.timeout,
            )
            assert_response_ok(response, "Failed to run select query:\n" + sparql_text)
            if verbose:
                print(response.text)
            if to_table:
                return pd.read_csv(StringIO(response.text), sep="\t", na_filter=False)
            else:
                return response.text

        else:
            raise NotImplementedError("Only RDFox engine is supported at the moment.")
    
    def add_data(self, data: str) -> None:
        """Add facts to the RDF database. Also can be used to update prefixes or add rules.

        Args:
            data (str): Data to add to the RDF database.

        Raises:
            NotImplementedError: Only RDFox engine is supported at the moment.
        """
        if self.engine == "rdfox":
            payload = {'operation': 'add-content-update-prefixes'}
            response = requests.patch(
                f"{self.endpoint}/datastores/{self.kg_name}/content", params=payload, data=data,
                timeout=self.timeout
            )
            assert_response_ok(response, "Failed to add facts to data store:\n" + data)
        else:
            raise NotImplementedError("Only RDFox engine is supported at the moment.")  
        
    def get_info(self, concept : str) -> Union[pd.DataFrame, None] :
        """Get the information of a given concept.
        The columns are: ?real_form, ?gibberish_form, ?pos, ?real_def, ?gibberish_def.

        Args:
            concept (str): Concept URI.

        Returns:
            Union[pd.DataFrame, None]: Information of the concept as a pandas DataFrame. None if the concept does not exist.
        """

        query = f"""
        SELECT DISTINCT (GROUP_CONCAT(DISTINCT STR(?real_forms) ; SEPARATOR=", ") AS ?real_form ) (GROUP_CONCAT(DISTINCT STR(?gibberish_forms) ; SEPARATOR=", ") AS ?gibberish_form) ?pos (STR(?real_def_t) AS ?real_def) (STR(?gibberish_def_t) AS ?gibberish_def) WHERE {{
            {concept} rdf:type ontolex:LexicalConcept ;
                sct:definition ?real_def_t ;
                sct:definition ?gibberish_def_t ;
                gwn:partOfSpeech ?pos ;
                sct:writtenForm ?real_forms ;
                sct:writtenForm ?gibberish_forms .

            FILTER ( 
                LANGMATCHES(LANG(?real_forms), "en") && LANGMATCHES(LANG(?real_def_t), "en") 
                && LANGMATCHES(LANG(?gibberish_forms), "fr") && LANGMATCHES(LANG(?gibberish_def_t), "fr")
            )
        }} GROUP BY ?pos ?real_def_t ?gibberish_def_t
        """
        answer_df = self.issue_query(query, to_table=True)
        if answer_df.shape[0] > 0:
            answer_df["?pos"] = answer_df["?pos"].str.split("wn#", expand=True)[1].str.slice(0, -1)
            return answer_df
        else:
            return None

    
    def check_word_exists(self, lemma : str, lang : str ="en") -> bool:
        query = f"""
        SELECT (COUNT(DISTINCT ?S) as ?C)
        WHERE {{
            ?S sct:writtenForm '{lemma}'@{lang} .
        }}
        """
        answer = self.issue_query(query, verbose=False)
        return int(answer.split("\n")[1]) > 0
    
    def get_concept_and_def(self, lemma : str, gwn_tag : str, lang="en"):
        """Get the concepts and definition of a given lemma and part of speech.

        Args:
            lemma (str): Lemma of the word.
            gwn_tag (str): GWN part of speech tag (e.g. "noun", "verb").
            lang (str, optional): Language of the lemma. Defaults to "en".

        Returns:
            str: Response to the SPARQL query.
        """
        
        query = f"""
        SELECT DISTINCT ?C ?D
        WHERE {{
            ?C sct:writtenForm "{escape_unicode(lemma)}"@{lang} .
            ?C gwn:partOfSpeech gwn:{gwn_tag} .
            ?C sct:definition ?D .
        }}
        """
        answer = self.issue_query(query, verbose=False)
        return answer
    
    def get_concept_with_gibberish(self, lemma : str, gwn_tag : str) -> str:
        """Get the concepts of a given lemma and part of speech in Gibberish.
        
        Args:
            lemma (str): Lemma of the gibberish word.
            gwn_tag (str): GWN part of speech tag (e.g. "noun", "verb").

        Returns:
            str: Response to the SPARQL query.
        """
        
        query = f"""
        SELECT DISTINCT ?C ?G
        WHERE {{
            ?C sct:writtenForm '{lemma}'@en .
            ?C gwn:partOfSpeech gwn:{gwn_tag} .
            ?C sct:writtenForm ?G .
            FILTER(LANGMATCHES (LANG (?G), "fr"))
        }}
        """
        answer = self.issue_query(query)
        return answer

    def add_gibberish_repr(self, gibberish_term : str, concepts : list[str]) -> None:
        """Add the representation of a gibberish term to some concepts in the RDF database.

        Args:
            gibberish_term (str): Gibberish term.
            concepts (list[str]): List of concept URIs to add the gibberish representation to.

        Raises:
            NotImplementedError: Only RDFox engine is supported at the moment.
        """

        # template_sense = "<https://en-word.net/lemma/unit#unit-oewn-00003553-n>"
        # template_entry = "<https://en-word.net/lemma/zwieback#zwieback-n>"
        # concept_template = "<https://en-word.net/id/oewn-00003553-n>"
        # template_cf = "<https://en-word.net/lemma/zwieback#zwieback-n-lemma>"

        triples = []

        for concept in concepts:
            id_end = concept.split("/id/")[-1]
            pos_concept = id_end[-2]
            gibberish_link = gibberish_term.replace(" ", "_")
            # We need to escape all the special characters such as à, é, è, ö, etc
            gibberish_link = unidecode.unidecode(gibberish_link)

            sense = f"<https://en-word.net/lemma/{gibberish_link}#{gibberish_link}-{id_end}"
            entry = f"<https://en-word.net/lemma/{gibberish_link}#{gibberish_link}{id_end[-3:]}"
            cf = f"<https://en-word.net/lemma/{gibberish_link}#{gibberish_link}-{pos_concept}-lemma>"
            # Lexical Sense
            triples.append(
                " ".join([sense, "<http://www.w3.org/ns/lemon/ontolex#isLexicalizedSenseOf>", concept , "."])
            )
            # Lexical Entry
            triples.append(
                " ".join([entry, "<http://www.w3.org/ns/lemon/ontolex#sense>", sense , "."])
            )
            # Canonical form
            triples.append(
                " ".join([entry, "<http://www.w3.org/ns/lemon/ontolex#canonicalForm>", cf , "."])
            )
            # Written form
            ## The gibberish term may contain special characters such as à, é, è, ö, etc that need to be escaped.
            # We use \u followed by the unicode code of the character to escape it.
            # For example, é becomes \\u00E9
            gibberish_term_ = escape_unicode(gibberish_term)
            triples.append(
                " ".join([cf, "<http://www.w3.org/ns/lemon/ontolex#writtenRep>", f'"{gibberish_term_}"@fr' , "."])
            )
            # Rdf types are already in the datalog of the triple store.
        
        triples =  "\n".join(triples)
        try:
            self.add_data(triples)
        except Exception as e:
            print(f"Failed to add gibberish representation {gibberish_term} of link {gibberish_link} to the RDF database.")
            print(triples)
            raise e
        
    def homonym_concepts(self, concept : str, lang : str ="en") -> str:
        """Get the homonym concepts of a given concept.

        Args:
            concept (str): Concept URI.
            lang (str, optional): Language of the homonym. Defaults to "en".

        Returns:
            str: Response to the SPARQL query.
        """
        query = f""" SELECT DISTINCT ?C ?H (STR(?Dd) as ?D) WHERE {{
            {concept} sct:writtenForm ?H .
            ?C sct:writtenForm ?H.
            ?C rdf:type ontolex:LexicalConcept .
            ?C sct:definition ?Dd .
            FILTER(
                (?C != {concept}) && LANGMATCHES (LANG(?H), "{lang}" ) && LANGMATCHES (LANG(?Dd), "{lang}" )
            )
        }}
        """
        return self.issue_query(query)
    
    def check_for_gibberish(self, concept : str) -> list[str]:
        """
        Check if a concept has a gibberish representation.

        Args:
            concept (str): Concept URI.
        
        Returns:
            list[str]: List of gibberish representations.
        """
        query = f"""
        SELECT (STR(DISTINCT ?F) AS ?G)
        WHERE {{
            {concept} sct:writtenForm ?F.
            {concept} rdf:type ontolex:LexicalConcept.
            FILTER(LANGMATCHES (LANG (?F), "fr"))
        }}
        """
        answer = self.issue_query(query)
        answer = answer[3:-1].replace('"', '')
        return [] if len(answer) == 0 else answer.split("\n")

    def check_gibberish_def(self, concept : str) -> list[str]:
        """Check if a concept has a gibberish definition.

        Args:
            concept (str): Concept URI.

        Returns:
            list[str]: List of gibberish definitions.
        """

        # Check if the concept already has a gibberish definition
        query = f"""
        SELECT (STR(DISTINCT ?F) AS ?G)
        WHERE {{
            {concept} sct:definition ?F.
            FILTER(LANGMATCHES (LANG(?F), "fr"))
        }}
        """
        answer = self.issue_query(query)
        answer = answer[3:-1].replace('"', '')
        return [] if len(answer) == 0 else answer.split("\n")
    
    def export(self, directory : Union[str, Path], title : str) -> None:
        """Export the RDF database to a file.

        Args:
            directory (Union[str, Path]): Directory to export the RDF database to.
            title (str): Title of the exported RDF database.
            format (str, optional): Format of the exported RDF database. Defaults to "ttl".

        Raises:
            NotImplementedError: Only RDFox engine is supported at the moment.
        """
        if self.engine == "rdfox":
            # Fact-domain is USER because we only want to export the facts that were added by the user.
            # The rules allow us to save some memory overhead by not exporting inferred facts.
            headers = {'Accept': 'text/turtle'}
            response = requests.get(
                f"{self.endpoint}/datastores/{self.kg_name}/content?fact-domain=explicit",
                headers=headers, timeout=self.timeout
            )
            assert_response_ok(response, "Failed to export the RDF database.")
            with open(Path(directory) / f"{title}.ttl", "w") as f:
                f.write(response.text)

            # Export rules as well
            headers = {'Accept': 'application/x.datalog'}
            response = requests.get(
                f"{self.endpoint}/datastores/{self.kg_name}/content?rule-domain=user",
                headers=headers, timeout=self.timeout
            )
            assert_response_ok(response, "Failed to export the rules.")
            with open(Path(directory) / f"{title}.dlog", "w") as f:
                f.write(response.text)
        else:
            raise NotImplementedError("Only RDFox engine is supported at the moment.")
        
