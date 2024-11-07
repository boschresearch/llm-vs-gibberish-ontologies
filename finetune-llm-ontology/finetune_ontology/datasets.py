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

# Import from huggingface
from datasets import Dataset
from transformers import pipeline
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
from random import seed, shuffle
from sklearn.model_selection import StratifiedKFold

# Path: finetune-llm-ontology/finetune_ontology/datasets.py

class PromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs):
        for variable in self.input_variables:
            if variable not in kwargs:
                raise ValueError(f"Variable {variable} is missing.")
        return self.template.format(**kwargs)

def make_prompt():
    template = """### HUMAN:
Identify whether the statement is true or false. Answer with only one word: 'True' or 'False'.

CONCEPT A: {gibberish_a} ({pos_a})
Definition: {definition_a}

CONCEPT B: {gibberish_b} ({pos_b})
Definition: {definition_b}

Statement: '{gibberish_a}' is a subclass of '{gibberish_b}'.

### ASSISTANT:
{label}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "gibberish_a", "pos_a", "definition_a",
            "gibberish_b", "pos_b", "definition_b",
            "label"
            ]
    )
    return prompt

def write_prompt(prompt : PromptTemplate, example,
                 label : Union[bool, str],
                 gibberish : bool):
    # Extract part of speech
    hypo_concept = example["?S"]
    hyper_concept = example["?T"]
    pos_dict = {
        "n": "noun",
        "v": "verb",
        "a": "adjective",
        "s": "adjective",
        "r": "adverb"
    }
    pos_a = pos_dict[hypo_concept[-2]]
    pos_b = pos_dict[hyper_concept[-2]]


    repr_hypo = example["?S_fg"] if gibberish else example["?S_f"]
    repr_hyper = example["?T_fg"] if gibberish else example["?T_f"]
    def_hypo = example["?S_def_g"] if gibberish else example["?S_def"]
    def_hyper = example["?T_def_g"] if gibberish else example["?T_def"]
    return prompt.format(
        gibberish_a=repr_hypo.replace(" | ", ", "),
        pos_a = pos_a,
        definition_a=def_hypo.replace("@fr", "").replace("@en", ""),
        gibberish_b=repr_hyper.replace(" | ", ", "),
        pos_b = pos_b,
        definition_b=def_hyper.replace("@fr", "").replace("@en", ""),
        label=label
    )

def write_negative_prompt(prompt : PromptTemplate, example,
                label : Union[bool, str],
                gibberish : bool):
    # Extract part of speech
    hypo_concept = example["?S"]
    hyper_concept = example["?T"]
    pos_dict = {
        "n": "noun",
        "v": "verb",
        "a": "adjective",
        "s": "adjective",
        "r": "adverb"
    }
    pos_a = pos_dict[hypo_concept[-2]]
    pos_b = pos_dict[hyper_concept[-2]]


    repr_hypo = example["?S_fg"] if gibberish else example["?S_f"]
    repr_hyper = example["?T_fg"] if gibberish else example["?T_f"]
    def_hypo = example["?S_def_g"] if gibberish else example["?S_def"]
    def_hyper = example["?T_def_g"] if gibberish else example["?T_def"]
    return prompt.format(
        gibberish_b=repr_hypo.replace(" | ", ", "),
        pos_b = pos_a,
        definition_b=def_hypo.replace("@fr", "").replace("@en", ""),
        gibberish_a=repr_hyper.replace(" | ", ", "),
        pos_a = pos_b,
        definition_a=def_hyper.replace("@fr", "").replace("@en", ""),
        label=False
    )

def make_hypernym_datasets(positive_path : Union[str, Path],
                           negative_path : Union[str, Path],
                           gibberish : bool,
                           validation : bool = False,
                           seed_val : int = 0,
                           folds : int = 5,):
    hypernym_dataset = pd.read_csv(positive_path, sep="\t", index_col=0)
    negative_dataset = pd.read_csv(negative_path, sep="\t", index_col=0)

    # Pick half of them
    train_concepts = hypernym_dataset["?S"].unique()[:len(hypernym_dataset["?S"].unique()) // 2]

    # train dataset
    train_dataset_positives = hypernym_dataset[hypernym_dataset["?S"].isin(train_concepts)]
    train_dataset_negatives = negative_dataset[negative_dataset["?S"].isin(train_concepts)]

    # test dataset
    test_dataset_positives = hypernym_dataset[~hypernym_dataset["?S"].isin(train_concepts)]
    test_dataset_negatives = negative_dataset[~negative_dataset["?S"].isin(train_concepts)]

    prompt = make_prompt()

    train_dataset = []
    for _, example in train_dataset_positives.iterrows():
        train_dataset.append(write_prompt(prompt, example, True, gibberish))
    # In order to add non trivial negatives, we will add the reverse one in the negatives.
        train_dataset.append(write_negative_prompt(prompt, example, False, gibberish))
    for _, example in train_dataset_negatives.iterrows():
        train_dataset.append(write_prompt(prompt, example, False, gibberish))
    print("Number of train positives :", train_dataset_positives.shape[0])
    print("Number of train negatives :", train_dataset_negatives.shape[0] + train_dataset_positives.shape[0])

    test_dataset = []
    n_pos = test_dataset_positives.shape[0]
    n_neg = test_dataset_negatives.shape[0]
    for _, example in test_dataset_positives.iterrows():
        test_dataset.append(write_prompt(prompt, example, True, gibberish))
    for _, example in test_dataset_negatives.iterrows():
        test_dataset.append(write_prompt(prompt, example, False, gibberish))

    if not validation:
        print("Number of test positives :", n_pos)
        print("Number of test negatives :", n_neg)

        seed(0)
        shuffle(train_dataset)

        train_dataset_hf = Dataset.from_dict({"text": train_dataset})
        test_dataset_hf = Dataset.from_dict({"text": test_dataset})

        return train_dataset_hf, test_dataset_hf
    else:
        kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed_val)
        # Perform k-fold cross validation
        ## Get all indices
        indices = np.arange(len(train_dataset))

        train_splits = []
        val_splits = []
        labels_ = np.array([1] * train_dataset_positives.shape[0] + [0] * (train_dataset_positives[0] + train_dataset_negatives.shape[0]))

        for j, (train_index, val_index) in enumerate(kf.split(indices, labels_)):
            train_splits.append(Dataset.from_dict({"text": [train_dataset[i] for i in train_index]}))
            val_splits.append(Dataset.from_dict({"text": [train_dataset[i] for i in val_index]}))
        
        test_dataset_hf = Dataset.from_dict({"text": test_dataset})

        return train_splits, val_splits, test_dataset_hf


def evaluate(model, tokenizer, test_dataset):
    model.eval()
    pipeline_ = pipeline("text-generation", model=model, tokenizer=tokenizer,
                         torch_dtype=model.config.torch_dtype,
                         device_map="auto")
    queries = [
        test_dataset["text"][i].split("### ASSISTANT:\n")[0] + '### ASSISTANT:\n' for i in range(len(test_dataset["text"]))
    ]
    labels = [
        test_dataset["text"][i].split("### ASSISTANT:\n")[-1] for i in range(len(test_dataset["text"]))
    ]
    labels = [1 if "true" in label.lower() else 0 for label in labels]


    sequences = pipeline_(
        queries,
        num_return_sequences=1,
        max_new_tokens=3,
        eos_token_id = tokenizer.eos_token_id,
        early_stopping=True,
        # do_sample = True
    )

    results = []
    for sequence, label in zip(sequences, labels):
        prediction = sequence[0]["generated_text"].split("### ASSISTANT:\n")[-1].strip()
        prediction = prediction.replace("\n", "")
        results.append(1 if "true" in prediction.lower() else 0)

    return queries, results, labels

    
    


    
    
