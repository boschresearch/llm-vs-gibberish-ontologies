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


from wordnet_gibberish.database import RDFDatabase
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm

def get_referenced_concepts(concept, rdfdb):
    query = f"""
    SELECT DISTINCT ?C (GROUP_CONCAT(DISTINCT STR(?real_forms) ; SEPARATOR=", ") AS ?real_form ) (GROUP_CONCAT(DISTINCT STR(?gibberish_forms) ; SEPARATOR=", ") AS ?gibberish_form) ?pos (STR(?real_def_t) AS ?real_def) (STR(?gibberish_def_t) AS ?gibberish_def)  WHERE {{
        ?C rdf:type ontolex:LexicalConcept ;
            sct:definition ?real_def_t ;
            sct:definition ?gibberish_def_t ;
            gwn:partOfSpeech ?pos ;
            sct:writtenForm ?real_forms ;
            sct:writtenForm ?gibberish_forms .
        
        {concept} sct:references ?C .

        FILTER (
            LANGMATCHES(LANG(?real_forms), "en") && LANGMATCHES(LANG(?real_def_t), "en") 
            && LANGMATCHES(LANG(?gibberish_forms), "fr") && LANGMATCHES(LANG(?gibberish_def_t), "fr")
        )

    }} GROUP BY ?C ?pos ?real_def_t ?gibberish_def_t
    """
    answer_df = rdfdb.issue_query(query, to_table=True)
    if answer_df.shape[0] == 0:
        return answer_df
    answer_df["?pos"] = answer_df["?pos"].str.split("wn#", expand=True)[1].str.slice(0, -1)
    return answer_df

def main(args):
    rdfdb = RDFDatabase(
        kg_name=f"WordNet-{args.which}",
        init_kg=False,
        add_prefixes=False
    )
    PATH_CONNECTED = Path(__file__).parent / "data" / "fake" / f"{args.which}_connected_concepts.csv"
    concepts_df = pd.read_csv(PATH_CONNECTED, sep="\t", index_col=0)
    concepts_df["connected"] = concepts_df["connected"].apply(eval)
    concepts_df["n_connections"] = concepts_df["connected"].apply(lambda x : sum([len(conn["related"]) for conn in x if conn["modify"]]))
    concepts_df["replacement_percentage"] = concepts_df.apply(lambda x : (
            get_referenced_concepts(x.name, rdfdb).shape[0] / x["n_connections"]
        ) if x["n_connections"] > 0 else 0,
        axis=1
    )
    concepts_df = concepts_df.sort_values(by=["replacement_percentage", "n_connections"], ascending=False)

    dataset = list(concepts_df.index)

    # Create new dataframe
    rows = []

    for concept in tqdm(dataset):
        x = rdfdb.get_info(concept)
        if x is None:
            continue
        x = x.iloc[0]
        repr = x["?real_form"].split(", ")
        x["?real_form"] = repr[0]

        repr = x["?gibberish_form"].split(", ")
        x["?gibberish_form"] = repr[0]
        x["?concept"] = concept
        rows.append(x.copy())
    
    # Make the dataframe
    df = pd.DataFrame(rows)
    df.to_csv(PATH_CONNECTED.parent / f"{args.which}_extraction_dataset.csv", sep="\t", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", type=str, default="sweets")
    args = parser.parse_args()
    main(args)
    



