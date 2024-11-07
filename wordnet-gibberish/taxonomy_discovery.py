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


import json
import argparse
from pathlib import Path
import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI
from langchain import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional, Union


### MODELS ###

class Triple(BaseModel):
    subject: str = "The subject (e.g. the subclass)"
    predicate: str = "The predicate (e.g. subClassOf)"
    object: str = "The object (e.g. the superclass)"

class HypernymyAnswer(BaseModel):
    concept_a : str = Field(description="The first concept.")
    concept_b : str = Field(description="The second concept.")
    full_answer : Optional[str] = Field(description="The full answer from the model.")
    thoughts: Optional[str] = Field(description="Write your justifications here.")
    answer : Optional[bool]

### UTILITIES ###

def parse_answer_json(answer):
    # parse ```json\n(.*?)``` in the answer
    start = answer.find("```json\n")
    answer_ = answer[start + len("```json\n"):]
    count = 3
    end = 0
    for i, c in enumerate(answer_):
        if i < 3:
            continue
        if c == "`":
            count -= 1
        if count == 0:
            end = i
            break
    if end == 0:
        print("[WARN] Could not extract json from following answer : ", answer)
        return None
    answer_ = answer_[:end-2]
    return answer_
    

def parse_answer_json_2(answer):
    # parse first json object in the answer that starts with { and ends with \n}
    
    # Get position of the first {
    start = answer.find("{")
    answer_ = answer[start:]
    count = 0
    # Add 1 to the count for each { and subtract 1 for each }. This assumes that there is no { or } in the textual fields.
    end = 0
    for i, c in enumerate(answer_):
        if c == "{":
            count += 1
        elif c == "}":
            count -= 1
        if count == 0:
            end = i
            break
    if end == 0:
        print("[WARN] Could not extract json from following answer : ", answer)
        return None
    answer_ = answer_[:end+1]
    return answer_

def shorten_uri(concept : str) -> str:
	"""Shorten the URI of a concept.

	Args:
		concept (str): Concept URI.

	Returns:
		str: Shortened concept URI.
	"""
	return f"id:{concept.split('/id/')[-1][:-1]}"

def read_hypernymy_dataset(path_positive : Union[str, Path], path_negative : Union[str, Path]) -> pd.DataFrame:
    positive = pd.read_csv(path_positive, sep="\t", index_col=0)
    positive["?S_def"] = positive["?S_def"].str.replace("@en", "")
    positive["?T_def"] = positive["?T_def"].str.replace("@en", "")
    positive["?S_def_g"] = positive["?S_def_g"].str.replace("@fr", "")
    positive["?T_def_g"] = positive["?T_def_g"].str.replace("@fr", "")
    negative = pd.read_csv(path_negative, sep="\t", index_col=0)
    negative["?S_def"] = positive["?S_def"].str.replace("@en", "")
    negative["?T_def"] = positive["?T_def"].str.replace("@en", "")
    negative["?S_def_g"] = positive["?S_def_g"].str.replace("@fr", "")
    negative["?T_def_g"] = positive["?T_def_g"].str.replace("@fr", "")
    print(positive.head(), negative.head())
    return positive, negative


def main(args):
    path_positives = Path(args.positives)
    path_negatives = Path(args.negatives)

    assert path_positives.exists(), f"Path {path_positives} does not exist."
    assert path_negatives.exists(), f"Path {path_negatives} does not exist."

    # Load dataset
    positive, negative = read_hypernymy_dataset(path_positives, path_negatives)

    # Create prompt
    answer_format = """```json
{
    "thoughts": "Reason step by step and write a concise reasoning here. This string can be empty.",
    "answer": true or false, but not both
}
```
"""
    if args.shots == "one":
        answer_example = """Example query:

CONCEPT A: potato
Definition: an edible tuber native to South America; a staple food of Ireland

CONCEPT B: root vegetable
Definition: any of various fleshy edible underground roots or tubers

Statement:
CONCEPT A is a subclass of CONCEPT B.

Expected answer:
```json
{
    "thoughts": "a potato is an edible tuber, which fits the definition of root vegetable.",
    "answer": true
}
```"""
    elif args.shots == "few":
        answer_example = """Here are a few examples that can aid you.

Example query:

CONCEPT A: potato
Definition: an edible tuber native to South America; a staple food of Ireland

CONCEPT B: root vegetable
Definition: any of various fleshy edible underground roots or tubers

Statement:
CONCEPT A is a subclass of CONCEPT B.

Expected answer:
```json
{
    "thoughts": "a potato is an edible tuber, which fits the definition of root vegetable.",
    "answer": true
}
```

Example query:

Concept A: music instrument
Definition: a device created or adapted for the purpose of making musical sounds

Concept B: animal
Definition: a living organism that feeds on organic matter, typically having specialized sense organs and nervous system and able to respond rapidly to stimuli

Statement:
Concept A is a subclass of Concept B.

Expected answer:
```json
{
    "thoughts": "A music instrument is not a living organism, hence it is not an animal.",
    "answer": false
}
```

Example query:

Concept A: car
Definition: a road vehicle, typically with four wheels, powered by an internal combustion engine and able to carry a small number of people

Concept B: vehicle
Definition: a thing used for transporting people or goods, especially on land, such as a car, truck, or cart

Statement:
Concept A is a subclass of Concept B.

Expected answer:
```json
{
    "thoughts": "A car is a vehicle, as it is a road vehicle.",
    "answer": true
}
```
"""
    else:
        answer_example = ""

    format_instructions = """Identify whether the statement is true or false.
Your response should only contain a JSON object, AND NO OTHER TEXT. The format is:\n""" + answer_format + answer_example
    
    template_question = """{format_instructions}
Query:

CONCEPT A: {writtenRep_a}
Definition: {definition_a}

CONCEPT B: {writtenRep_b}
Definition: {definition_b}

Statement:
CONCEPT A is a subclass of CONCEPT B.
"""

    prompt_template = PromptTemplate(
        input_variables = ["format_instructions", "writtenRep_a", "definition_a", "writtenRep_b", "definition_b"],
        template = template_question,
    )

    # Load model
    load_dotenv()
    LLM = AzureChatOpenAI(
        openai_api_version="2023-03-15-preview",
        azure_deployment="gpt35",
        openai_api_type="azure",
        temperature=0.0,
    )

    def predict(x):
        concept_a = x["?S_f"].replace(" | ", ", ")
        definition_a = x["?S_def"]
        concept_b = x["?T_f"].replace(" | ", ", ")
        definition_b = x["?T_def"]

        conversation = LLMChain(
            prompt = prompt_template,
            llm=LLM,
            verbose=False
        )

        try:
            answer = conversation.predict(
                format_instructions=format_instructions,
                writtenRep_a = concept_a,
                definition_a = definition_a,
                writtenRep_b = concept_b,
                definition_b = definition_b
            )
            try:
                # Parse answer
                answer_ = parse_answer_json(answer)
                if answer_ is None:
                    answer_ = parse_answer_json_2(answer)

                parsed_answer_real = HypernymyAnswer(
                    concept_a=concept_a,
                    concept_b=concept_b,
                    full_answer=answer,
                    **json.loads(answer_)
                )
            except Exception as e:
                print(f"[ERROR] Could not parse the result for {concept_a} and {concept_b}.")
                print(e)
                parsed_answer_real = HypernymyAnswer(
                    concept_a=concept_a,
                    concept_b=concept_b,
                    full_answer=answer,
                    thoughts= "Could not parse the result",
                    answer=None
                )
        except:
            print(f"[ERROR] Could not get an answer for {concept_a} and {concept_b}.")
            print(e)
            parsed_answer_real = HypernymyAnswer(
                concept_a=concept_a,
                concept_b=concept_b,
                full_answer=str(e),
                thoughts= "Could not parse the result",
                answer=None
            )
        
        concept_a = x["?S_fg"].replace(" | ", ", ")
        definition_a = x["?S_def_g"]
        concept_b = x["?T_fg"].replace(" | ", ", ")
        definition_b = x["?T_def_g"]
        
        conversation = LLMChain(
            prompt = prompt_template,
            llm=LLM,
            verbose=False
        )

        try:
            answer = conversation.predict(
                format_instructions=format_instructions,
                writtenRep_a = concept_a,
                definition_a = definition_a,
                writtenRep_b = concept_b,
                definition_b = definition_b
            )
            try:
                # Parse answer
                answer_ = parse_answer_json(answer)
                if answer_ is None:
                    answer_ = parse_answer_json_2(answer)

                parsed_answer_fake = HypernymyAnswer(
                    concept_a=concept_a,
                    concept_b=concept_b,
                    full_answer=answer,
                    **json.loads(answer_)
                )
            except Exception as e:
                print(f"[ERROR] Could not parse the result for {concept_a} and {concept_b}.")
                print(e)
                parsed_answer_fake = HypernymyAnswer(
                    concept_a=concept_a,
                    concept_b=concept_b,
                    full_answer=answer,
                    thoughts= "Could not parse the result",
                    answer=None
                )
        except:
            print(f"[ERROR] Could not get an answer for {concept_a} and {concept_b}.")
            print(e)
            parsed_answer_fake = HypernymyAnswer(
                concept_a=concept_a,
                concept_b=concept_b,
                full_answer=str(e),
                thoughts= "Could not parse the result",
                answer=None
            )

        return parsed_answer_real, parsed_answer_fake
    
    # Extract triples

    output_dir = Path(args.save_dir)
    output_pos = output_dir / "positive"
    output_neg = output_dir / "negative"
    output_pos.mkdir(exist_ok=True, parents=True)
    output_neg.mkdir(exist_ok=True, parents=True)

    for i, row in tqdm(positive.iterrows(), total=len(positive)):
        if not( (output_pos / f"{i}_real.json").exists() and (output_pos / f"{i}_fake.json").exists() ):

            answer_real, answer_fake = predict(row)

            with open(output_pos / f"{i}_real.json", "w") as f:
                json.dump(
                    answer_real.dict(),
                    f,
                    indent=4
                )
            with open(output_pos / f"{i}_fake.json", "w") as f:
                json.dump(
                    answer_fake.dict(),
                    f,
                    indent=4
                )
    
    for i, row in tqdm(negative.iterrows(), total=len(negative)):
        if not( (output_neg / f"{i}_real.json").exists() and (output_neg / f"{i}_fake.json").exists() ):

            answer_real, answer_fake = predict(row)

            with open(output_neg / f"{i}_real.json", "w") as f:
                json.dump(
                    answer_real.dict(),
                    f,
                    indent=4
                )
            with open(output_neg / f"{i}_fake.json", "w") as f:
                json.dump(
                    answer_fake.dict(),
                    f,
                    indent=4
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--positives", type=str, help="Path to the positive hypernymy dataset.")
    parser.add_argument("--negatives", type=str, help="Path to the negative hypernymy dataset.")
    parser.add_argument("--shots", type=str, help="Whether to provide one or few examples.", default="one")
    parser.add_argument("--save_dir", type=str, help="Path to the directory where the results will be saved.")
    args = parser.parse_args()
    main(args)
    
        








