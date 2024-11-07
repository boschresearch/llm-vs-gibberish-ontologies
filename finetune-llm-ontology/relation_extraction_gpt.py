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
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional


### MODELS ###

class Triple(BaseModel):
    subject: str = "The subject (e.g. the subclass)"
    predicate: str = "The predicate (e.g. subClassOf)"
    object: str = "The object (e.g. the superclass)"

class ExtractedOntology(BaseModel):
    full_answer: Optional[str] = Field(description="The full answer from the model.")
    thoughts: str = Field(description="Write your justifications here.")
    triples: list[Triple] = [
        Triple(subject="subclass", predicate="subClassOf", object="superclass"),
        Triple(subject="part", predicate="partOf", object="whole")
    ]

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


### MAIN SCRIPT ###

def main(args):
    path_extraction_dataset = Path(args.extraction_dataset)

    # Load dataset
    extraction_df = pd.read_csv(path_extraction_dataset, sep='\t')

    # Load model
    load_dotenv()
    if args.model == "gpt35":
        LLM = AzureChatOpenAI(
            openai_api_version="2023-03-15-preview",
            azure_deployment="gpt35",
            openai_api_type="azure",
            temperature=0.0,
        )
    elif args.model == "gpt4":
        LLM = AzureChatOpenAI(
            openai_api_version="2023-03-15-preview",
            azure_deployment="gpt4",
            openai_api_type="azure",
            temperature=0.0,
        )
    elif args.model == "dbrx":
        LLM = ChatOpenAI(
            model_name = "databricks/dbrx-instruct",
            default_headers={"api-key": os.environ.get("api_key")},
            openai_api_base=os.environ.get("api_base"), 
            openai_api_key=os.environ.get("api_key"),
            openai_proxy="http://localhost:3128",
            max_tokens=800,
            temperature=0.0,
        )
    elif args.model == "zephyr":
        LLM = ChatOpenAI(
            model_name = "HuggingFaceH4/zephyr-7b-beta",
            default_headers={"api-key": os.environ.get("api_key")},
            openai_api_base=os.environ.get("api_base"), 
            openai_api_key=os.environ.get("api_key"),
            openai_proxy="http://localhost:3128",
            max_tokens=800,
            temperature=0.0,
        )

    # Define prompt
    answer_format ="""```json
{
    "thoughts": "Reason step by step and write your justifications here.",
    "triples" : [
        {"subject" : "subclass", "predicate": "is a subclass of", "object": "superclass"},
        {"subject" : "part", "predicate": "is a component of", "object": "whole"}
    ]
}
```
"""

    answer_example = """
Example query:

Concept name: carrot juice
Part-of-speech: noun
Definition: Juice made out of carrot

Expected answer (with NOTHING ELSE):
```json
{
    "thoughts": "The definition states that 'carrot juice' is a 'juice', hence 'carrot juice' is a subclass of 'juice'. Moreover, the definition mentions that 'carrot juice' is made from 'carrot', hence 'carrot' is a component of 'carrot juice'.",
    "triples": [
        {"subject" : "carrot juice", "predicate": "is a subclass of", "object": "juice"},
        {"subject" : "carrot", "predicate": "is a component of", "object": "carrot juice"}
    ]
}
```
"""

    template_example = """The input data contains a concept, its part-of-speech tagging and its definition. It is expected that you may not know some of those words.
Extract a taxonomy of ALL the terms used in the definition, and only those terms (including the defined concept). Only use the relations: "is a subclass" of and "is a component of" .
Your response should only contain a JSON object, AND NO OTHER TEXT. The format is:\n""" + answer_format + answer_example

    template_question = """{format_instructions}
INPUT DATA :
Concept name: {writtenRep}
Part-of-speech: {pos}
Definition: {definition}
"""

    prompt_template = PromptTemplate(
        input_variables = ["format_instructions", "writtenRep", "definition", "pos"],
        template = template_question
    )

    # Function to get answers

    def get_answers(concept_info, prompt_template, template_example : str = template_example, remove_real_concept : bool = False) -> tuple[ExtractedOntology]:
        conversation = conversation = LLMChain(
        prompt = prompt_template,
        llm=LLM,
        verbose=False
        )
        answer_real = conversation.predict(format_instructions = template_example, writtenRep = "X" if remove_real_concept else concept_info["?real_form"], definition= concept_info["?real_def"], pos=concept_info["?pos"])
        answer_real_ = parse_answer_json(answer_real)
        if answer_real_ is None:
            answer_real_ = parse_answer_json_2(answer_real)

        try:
            answer_real = ExtractedOntology(
                full_answer=answer_real,
                **json.loads(answer_real_)
            )
        except:
            answer_real = ExtractedOntology(
                full_answer=answer_real,
                thoughts= f"Could not parse the result",
                triples=[]
            )

        conversation = LLMChain(
        prompt = prompt_template,
        llm=LLM,
        verbose=False
        )
        answer_gibberish = conversation.predict(format_instructions = template_example, writtenRep = concept_info["?gibberish_form"], definition= concept_info["?gibberish_def"], pos=concept_info["?pos"])
        answer_gibberish_ = parse_answer_json(answer_gibberish)
        if answer_gibberish_ is None:
            answer_gibberish_ = parse_answer_json_2(answer_gibberish)
        try:
            answer_gibberish = ExtractedOntology(
                full_answer=answer_gibberish,
                **json.loads(answer_gibberish_)
            )
        except:
            answer_gibberish = ExtractedOntology(
                full_answer=answer_gibberish,
                thoughts= f"Could not parse the result",
                triples=[]
            )
        return answer_real, answer_gibberish

    # Extract triples
    answers_real = dict()
    answers_fake = dict()
    concepts = dict()

    save_dir = args.save_dir
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)


    for i, x in tqdm(extraction_df.iterrows(), total=extraction_df.shape[0]):
        answer_real, answer_gibberish = get_answers(x, prompt_template)

        answers_real[i] = answer_real.copy(deep=True)
        answers_fake[i] = answer_gibberish.copy(deep=True)
        concepts[i] = x["?concept"]

        with open(save_dir / f"{shorten_uri(concepts[i])}_real.json", "w") as f:
            json.dump(
                answers_real[i].dict(),
                f,
                indent=4
            
            )
        with open(save_dir / f"{shorten_uri(concepts[i])}_fake.json", "w") as f:
            json.dump(
                answers_fake[i].dict(),
                f,
                indent=4
            )
    
    # Save results
    for i in concepts.keys():
        with open(save_dir / f"{shorten_uri(concepts[i])}_real.json", "w") as f:
            json.dump(
                answers_real[i].dict(),
                f,
                indent=4
            
            )
        with open(save_dir / f"{shorten_uri(concepts[i])}_fake.json", "w") as f:
            json.dump(
                answers_fake[i].dict(),
                f,
                indent=4
            )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract relations from a dataset.")
    parser.add_argument("--extraction_dataset", type=str, required=True, help="Path to the dataset to extract relations from.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the extracted relations.")
    parser.add_argument("--model", type=str, default="gpt35", help="Model to use for the extraction.")
    args = parser.parse_args()
    main(args)