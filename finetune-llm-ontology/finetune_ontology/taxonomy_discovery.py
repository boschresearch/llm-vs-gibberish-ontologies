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
from .relation_extraction import parse_answer_json, parse_answer_json_2
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

class HypernymyAnswer(BaseModel):
    concept_a : str = Field(description="The first concept.")
    concept_b : str = Field(description="The second concept.")
    full_answer : Optional[str] = Field(description="The full answer from the model.")
    thoughts: Optional[str] = Field(description="Write your justifications here.")
    answer : Optional[bool]

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

def make_hypernymy_prompt(model_name : str):
    if "llama" in model_name:
        answer_format = """```json
{
    "thoughts": "Reason step by step and write your justifications in one line here. This string can be empty.",
    "answer": true or false, but not both
}
```
"""

        answer_example = """
    Example query:

    CONCEPT A: potato
    Definition: an edible tuber native to South America; a staple food of Ireland

    CONCEPT B: root vegetable
    Definition: any of various fleshy edible underground roots or tubers

    Expected answer:
    ```json
    {
        "thoughts": "a potato is an edible tuber, which fits the definition of root vegetable.",
        "answer": true
    }
    ```
"""

        template_example = """
Identify whether CONCEPT A is a subclass of CONCEPT B.
Your response should only contain one single JSON object, AND NOTHING ELSE. You must use the following format:\n""" + answer_format + answer_example

        template_question = """### HUMAN :
{format_instructions}
Query:

CONCEPT A: {writtenRep_a}
Definition: {definition_a}

CONCEPT B: {writtenRep_b}
Definition: {definition_b}

### ASSISTANT:
Answer:
```json
"""

    else:
        answer_format = """```json
{
    "thoughts": "Reason step by step and write your justifications in one line here. This string can be empty.",
    "answer": true or false, but not both
}
```
"""

        answer_example = """
Example query:

CONCEPT A: potato
Definition: an edible tuber native to South America; a staple food of Ireland

CONCEPT B: root vegetable
Definition: any of various fleshy edible underground roots or tubers

Expected answer:
```json
{
    "thoughts": "a potato is an edible tuber, which fits the definition of root vegetable.",
    "answer": true
}
```
"""

        template_example = """
Identify whether CONCEPT A is a subclass of CONCEPT B.
Your response should only contain one single JSON object, AND NOTHING ELSE. You must use the following format:\n""" + answer_format + answer_example

        template_question = """### HUMAN :
{format_instructions}
Query:

CONCEPT A: {writtenRep_a}
Definition: {definition_a}

CONCEPT B: {writtenRep_b}
Definition: {definition_b}

### ASSISTANT:
Answer:
```json
"""

    prompt_template = PromptTemplate(
        input_variables = ["format_instructions", "writtenRep_a", "definition_a", "writtenRep_b", "definition_b"],
        template = template_question
    )
    return prompt_template, template_example

def predict(pipeline, prompts : list[str], eos_token_id):
    if eos_token_id is None or (isinstance(eos_token_id, list) and len(eos_token_id) == 0):
        res = pipeline(
            prompts,
            num_return_sequences=1,
            min_new_tokens=100,
            max_new_tokens=500,
            early_stopping=True,
            do_sample=False
        )
    else:
        res = pipeline(
            prompts,
            num_return_sequences=1,
            min_new_tokens=100,
            max_new_tokens=500,
            early_stopping=True,
            eos_token_id=eos_token_id,
            do_sample=False
        )
    res = [ res[i][0]["generated_text"].split('### ASSISTANT:\nAnswer:\n')[1] for i in range(len(res))]
    return res

def get_answers(
        x,
        prompt_template : PromptTemplate,
        template_example : str,
        pipeline : TextGenerationPipeline,
        parse_before_return : bool = True,
        eos_token_id = None,
        verbose : bool = False
):
    concept_a = x["?S_f"].replace(" | ", ", ")
    definition_a = x["?S_def"]
    concept_b = x["?T_f"].replace(" | ", ", ")
    definition_b = x["?T_def"]
    prompt = prompt_template.format(
        format_instructions=template_example,
        writtenRep_a=concept_a,
        definition_a=definition_a,
        writtenRep_b=concept_b,
        definition_b=definition_b
    )
    res = predict(pipeline, [prompt], eos_token_id)[0]
    if verbose:
        print(res)
    
    if parse_before_return:
        answer_real = parse_answer_json(res)
        if answer_real is None:
            answer_real = parse_answer_json_2(res)
    else:
        answer_real = res
    
    try:
        answer_real = HypernymyAnswer(
            concept_a=concept_a,
            concept_b=concept_b,
            full_answer=res,
            **json.loads(answer_real)
        )
    except:
        print("[ERROR] Could not parse the result : ", prompt, answer_real)
        answer_real = HypernymyAnswer(
            concept_a=concept_a,
            concept_b=concept_b,
            full_answer=res,
            thoughts="Could not parse the result",
            answer=None
        )
    
    # Fake answer

    concept_a = x["?S_fg"].replace(" | ", ", ")
    definition_a = x["?S_def_g"]
    concept_b = x["?T_fg"].replace(" | ", ", ")
    definition_b = x["?T_def_g"]

    prompt = prompt_template.format(
        format_instructions=template_example,
        writtenRep_a=concept_a,
        definition_a=definition_a,
        writtenRep_b=concept_b,
        definition_b=definition_b
    )

    res = predict(pipeline, [prompt], eos_token_id)[0]
    if verbose:
        print(res)

    if parse_before_return:
        answer_fake = parse_answer_json(res)
        if answer_fake is None:
            answer_fake = parse_answer_json_2(res)
    else:
        answer_fake = res

    try:
        answer_fake = HypernymyAnswer(
            concept_a=concept_a,
            concept_b=concept_b,
            full_answer=res,
            **json.loads(answer_fake)
        )
    except:
        print("[ERROR] Could not parse the result : ", prompt, answer_fake)
        answer_fake = HypernymyAnswer(
            concept_a=concept_a,
            concept_b=concept_b,
            full_answer=res,
            thoughts="Could not parse the result",
            answer=None
        )
    return answer_real, answer_fake


def pipeline_discovery(model_name : str,
                    positive_path : Union[str, Path],
                    negative_path : Union[str, Path],
                    pipeline : TextGenerationPipeline,
                    eos_token_id : None,
                    save_path : Union[str, Path],
                    skip_if_exists : bool = True,
                    verbose : bool = False):
    # Read the dataset
    positive, negative = read_hypernymy_dataset(positive_path, negative_path)
    # Make the prompt
    prompt_template, template_example = make_hypernymy_prompt(model_name)

    (save_path / "positive").mkdir(parents=True, exist_ok=True)
    (save_path / "negative").mkdir(parents=True, exist_ok=True) 

    # Get the answers
    for i, x in tqdm(positive.iterrows(), total=len(positive)):
        if skip_if_exists and (save_path / "positive" / f"{i}_real.json").exists() and (save_path / "positive" / f"{i}_fake.json").exists():
            continue
        answer_real, answer_fake = get_answers(x, prompt_template, template_example, pipeline, True, eos_token_id, verbose)
        # Save the answers
        with open(save_path / "positive" / f"{i}_real.json", "w") as f:
            json.dump(answer_real.dict(), f, indent=4)
        with open(save_path / "positive" / f"{i}_fake.json", "w") as f:
            json.dump(answer_fake.dict(), f, indent=4)
    
    for i, x in tqdm(negative.iterrows(), total=len(negative)):
        if skip_if_exists and (save_path / "negative" / f"{i}_real.json").exists() and (save_path / "negative" / f"{i}_fake.json").exists():
            continue
        answer_real, answer_fake = get_answers(x, prompt_template, template_example, pipeline, True, eos_token_id, verbose)
        # Save the answers
        with open(save_path / "negative" / f"{i}_real.json", "w") as f:
            json.dump(answer_real.dict(), f, indent=4)
        with open(save_path / "negative" / f"{i}_fake.json", "w") as f:
            json.dump(answer_fake.dict(), f, indent=4)


    