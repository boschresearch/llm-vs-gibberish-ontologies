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

from .datasets import PromptTemplate
from pydantic import BaseModel
from pydantic.fields import Field
from typing import Optional, Union
from pathlib import Path
import pandas as pd
from transformers import TextGenerationPipeline
import json
from tqdm import tqdm
import re


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

def read_extraction_dataset(path : Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", na_filter=False)

def make_extraction_prompt(model_name : str):
    answer_example = """
Example query:

Concept name: carrot juice
Part-of-speech: noun
Definition: Juice made out of carrot

Expected answer:
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
    if "llama" in model_name:


        answer_format = """```json
{
    "thoughts": "Reason step by step and write your justifications here.",
    "triples" : [
        {"subject" : "subclass", "predicate": "is a subclass of", "object": "superclass"},
        {"subject" : "part", "predicate": "is a component of", "object": "whole"}
    ]
}
```
"""
        template_example = """The input data contains a concept, its part-of-speech tagging and its definition. It is expected that you may not know some of those words.
Extract a taxonomy of ALL the terms used in the definition, and only those terms (including the defined concept). Only use the relations: "is a subclass" of and "is a component of" .
Your response should only contain a JSON object, AND NO OTHER TEXT. The format is:\n""" + answer_format + answer_example
        
        template_question = """### HUMAN :
{format_instructions}
INPUT DATA :
Concept name: {writtenRep}
Part-of-speech: {pos}
Definition: {definition}

### ASSISTANT:
Answer:
"""
    else:
        answer_format = """```json
{
    "thoughts": "Reason step by step and write a concise justification here.",
    "triples" : [
        {"object" : "subclass", "predicate": "is a subclass of", "object": "superclass"},
        {"subject" : "part", "predicate": "is a component of", "object": "whole"}
    ]
}
```
"""
        template_example = """Extract a taxonomy of ALL the terms used in the definition, and only those terms (including the defined concept). Only use the relations: "is a subclass" of and "is a component of" .
Your response should only contain a JSON object, AND NO OTHER TEXT. The format is:\n""" + answer_format + answer_example

        template_question = """### HUMAN :
{format_instructions}
Query:

Concept name: {writtenRep}
Part-of-speech: {pos}
Definition: {definition}

### ASSISTANT:
Answer:
"""


    prompt_template = PromptTemplate(
        input_variables = ["format_instructions", "writtenRep", "definition", "pos"],
        template = template_question
    )
    return prompt_template, template_example

def predict(pipeline, prompts : list[str], eos_token_id):
    res = pipeline(
        prompts,
        num_return_sequences=1,
        min_new_tokens=100,
        max_new_tokens=500,
        early_stopping=True,
        do_sample=False
    )
    res = [ res[i][0]["generated_text"].split('### ASSISTANT:\nAnswer:\n')[1] for i in range(len(res))]
    return res

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

def get_answers(concept_info, prompt_template, template_example : str,
                pipeline : TextGenerationPipeline,
                parse_before_return : bool = True,
                remove_real_concept : bool = False,
                eos_token_id = None,
                verbose : bool = False
                ) -> tuple[ExtractedOntology]:
    prompt = prompt_template.format(
        format_instructions=template_example,
        writtenRep=concept_info["?real_form"],
        definition=concept_info["?real_def"],
        pos=concept_info["?pos"]
    )
    full_answer_real = predict(pipeline, [prompt], eos_token_id)[0]
    if verbose:
        print(full_answer_real)

    if parse_before_return:
        answer_real = parse_answer_json(full_answer_real)
    else:
        answer_real = full_answer_real
    try:
        answer_real = ExtractedOntology(
            full_answer = full_answer_real,
            **json.loads(answer_real)
        )
    except:
        answer_real = ExtractedOntology(
            full_answer=full_answer_real,
            thoughts= f"Could not parse the result : {answer_real}",
            triples=[]
        )

    prompt = prompt_template.format(
        format_instructions=template_example,
        writtenRep=concept_info["?gibberish_form"],
        definition=concept_info["?gibberish_def"],
        pos=concept_info["?pos"]
    )
    full_answer_gibberish = predict(pipeline, [prompt], eos_token_id)[0]
    if verbose:
        print(full_answer_gibberish)

    if parse_before_return:
        answer_gibberish = parse_answer_json(full_answer_gibberish)
    else:
        answer_gibberish = full_answer_gibberish
    try:
        answer_gibberish = ExtractedOntology(
            full_answer=full_answer_gibberish,
            **json.loads(answer_gibberish)
        )
    except:
        answer_gibberish = ExtractedOntology(
            full_answer=full_answer_gibberish,
            thoughts= f"Could not parse the result : {answer_gibberish}",
            triples=[]
        )
    return answer_real, answer_gibberish

def shorten_uri(concept : str) -> str:
    """Shorten the URI of a concept.

    Args:
        concept (str): Concept URI.

    Returns:
        str: Shortened concept URI.
    """
    return f"id:{concept.split('/id/')[-1][:-1]}"

def pipeline_extraction(model_name : str,
                        path : Union[str, Path],
                        pipeline : TextGenerationPipeline,
                        eos_token_id,
                        save_path : Union[str, Path],
                        verbose : bool = True):
    concept_list = []
    answers_real = []
    answers_fake = []

    SAVE_PATH = Path(save_path)

    prompt_template, template_example = make_extraction_prompt(model_name)
    df = read_extraction_dataset(path)
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        answer_real, answer_fake = get_answers(
            row, prompt_template, template_example, pipeline, eos_token_id, verbose=verbose
        )
        answers_real.append(answer_real.model_copy(deep=True))
        answers_fake.append(answer_fake.model_copy(deep=True))
        c = row["?concept"]
        file_name = f"{shorten_uri(c)}_real.json"
        with open(SAVE_PATH / file_name, "w") as f:
            f.write(answers_real[i].model_dump_json(indent=4))
        file_name = f"{shorten_uri(c)}_fake.json"
        with open(SAVE_PATH / file_name, "w") as f:
            f.write(answers_fake[i].model_dump_json(indent=4))

        concept_list.append(row["?concept"])

    return concept_list, answers_real, answers_fake

        
