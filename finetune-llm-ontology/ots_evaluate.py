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

import argparse
import requests
from pathlib import Path
import json
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score
from swiplserver import PrologMQI, PrologThread

from wordnet_gibberish.database import RDFDatabase
from wordnet_gibberish.utils.utils import shorten_uri, escape_unicode


### RELATION EXTRACTION ###
def is_subclass_of(written_form_a, written_form_b, lang="en", rdfdb : Optional[RDFDatabase] = None):
    query = f"""SELECT DISTINCT ?A ?B WHERE {{
        ?A rdf:type ontolex:LexicalConcept ;
            sct:writtenForm ?F_a .
        
        ?B rdf:type ontolex:LexicalConcept ;
            sct:writtenForm ?F_b .
        
        ?A sct:hypernym ?B .

        FILTER(
            (STR(?F_a) = "{written_form_a}") && (STR(?F_b) = "{written_form_b}")
            {f"&& LANGMATCHES(LANG(?F_a), '{lang}') && LANGMATCHES(LANG(?F_b), '{lang}')" if lang else ""}
        )
    }}
"""
    answer = rdfdb.issue_query(query, to_table=True)
    return answer.shape[0] > 0, answer

def positive_pairs(concept, lang="en", rdfdb : Optional[RDFDatabase] = None, verbose=False):
    # Get the definition of the concept.
    query = f"""
    SELECT (GROUP_CONCAT(DISTINCT STR(?F); SEPARATOR=" | ") AS ?forms) ?D WHERE {{
        {concept} rdf:type ontolex:LexicalConcept ;
            sct:writtenForm ?F ;
            sct:definition ?D .

        FILTER(
            LANGMATCHES(LANG(?F), "{lang}") && LANGMATCHES(LANG(?D), "{lang}")
        )
    }} GROUP BY ?D
"""

    x = rdfdb.issue_query(query, to_table=True).iloc[0]

    forms = x["?forms"].split(" | ")
    definition = x["?D"]

    if verbose:
        print(x)

    # Get the hypernyms within the definition

    query = f""" SELECT DISTINCT (STR(?F) AS ?HYPONYM) (STR(?G) AS ?HYPERNYM) WHERE {{
    
        ?hypo_concept rdf:type ontolex:LexicalConcept ;
            sct:writtenForm ?F ;
            sct:definition ?D .

        ?hyper_concept rdf:type ontolex:LexicalConcept ;
            sct:writtenForm ?G ;
            sct:definition ?E .
            

        FILTER(
            ( CONTAINS("{definition}", STR(?F)) || (STR(?F) = "{forms[0]}") )
            && ( CONTAINS("{definition}", STR(?G)) || (STR(?G) = "{forms[0]}") )
        )

        ?hypo_concept sct:hypernym ?hyper_concept .

    }}"""
    answer = rdfdb.issue_query(query, to_table=True)
    return answer

def transitive_closure(answer, path_program = Path(__file__) / "prolog_programs"):  
    # Write prolog program
    prolog_program = ""
    n_asserts = 0
    for triple in answer.triples:
        if triple.predicate == "is a subclass of":
            prolog_program += f"""subClassOf("{triple.subject}", "{triple.object}").\n"""
            n_asserts += 1
    
    if n_asserts == 0:
        return set()
    
    # add rule for transitive closure:
    prolog_program += """ancestor(X, Y) :- ancestor(X, Y, []).
ancestor(X, Y, A) :- subClassOf(X, Y), \+ memberchk((X, Y), A).
ancestor(X, Y, A) :- subClassOf(X, Z), \+ memberchk((X, Z), A), ancestor(Z, Y, [(X, Z)|A]).
"""
    # Save program
    with open(path_program / "prolog_program.pl", "w") as f:
        f.write(prolog_program)
    
    # Solve the program
    try:
        with PrologMQI() as mqi:
            with mqi.create_thread() as prolog_thread:
                prolog_thread.query(f"consult('{path_program / 'prolog_program.pl'}')")
                result = prolog_thread.query("ancestor(X, Y)")
        answer_set = set()
        for r in result:
            answer_set.add((r["X"], r["Y"]))
        return answer_set
    except:
        print(prolog_program)
        raise Exception("Prolog program failed.")
    
def get_metrics(concept, answer, closure=True, lang="en", rdfdb : Optional[RDFDatabase] = None):
    positive_pairs_ = positive_pairs(concept, lang=lang, rdfdb=rdfdb)
    positive_set = set()
    for i, row in positive_pairs_.iterrows():
        positive_set.add((row["?HYPONYM"], row["?HYPERNYM"]))

    if closure:
        answer_set = transitive_closure(answer)
    else:
        answer_set = set()
        for triple in answer.triples:
            if triple.predicate == "is a subclass of":
                answer_set.add((triple.subject, triple.object))
    
    tp_set = positive_set.intersection(answer_set)
    fp_set = answer_set.difference(positive_set)
    fn_set = positive_set.difference(answer_set)

    return tp_set, fp_set, fn_set

def correspondence_wrt_real(query_concept, real_answer, fake_answer, rdfdb):
    tp = 0 # Present in both real and fake
    fn = 0 # Present in real but not in fake
    fp = 0 # Present in fake but not in real

    # TODO: deal with this
    accepted_predicates = set([
        "is a subclass of", "is a component of"
    ])

    triples_real_converted = set()
    triples_fake = set()

    for triple in fake_answer.triples:
        if triple.predicate in accepted_predicates:
            triples_fake.add(
                (triple.subject, triple.predicate, triple.object)
            )
    
    for triple in real_answer.triples:
        subject = triple.subject
        predicate = triple.predicate
        object = triple.object

        if not predicate in accepted_predicates:
            continue
        
        # Convert

        query_convert = f"""
        SELECT DISTINCT (STR(?Fg) AS ?F) ?Dg WHERE {{
            ?C rdf:type ontolex:LexicalConcept .
            ?C sct:writtenForm "{escape_unicode(subject)}"@en .
            ?C sct:writtenForm ?Fg .
            FILTER (
                LANGMATCHES(LANG(?Fg), "fr")
            )
            OPTIONAL {{
                {query_concept} sct:definition ?Dg .
                FILTER(
                    CONTAINS(STR(?Dg), STR(?Fg)) && LANGMATCHES(LANG(?Dg), "fr")
                )
            }}
        }}
        """
        
        subject_converted = rdfdb.issue_query(query_convert, to_table=True)
        if subject_converted.shape[0] > 0:
            if subject_converted.shape[0] > 1:
                # print(f"[WARN] There is more than 1 gibberish representation for {subject}.")
                subject_converted_ = subject_converted.dropna(how="any", axis=0)
            else:
                subject_converted_ = subject_converted
            subject_converted = subject_converted_["?F"].iloc[0] if subject_converted_.shape[0] > 0 else subject_converted["?F"].iloc[0]
        else:
            subject_converted = subject
        
        query_convert = f"""
        SELECT DISTINCT (STR(?Fg) AS ?F) ?Dg WHERE {{
            ?C rdf:type ontolex:LexicalConcept .
            ?C sct:writtenForm "{escape_unicode(object)}"@en .
            ?C sct:writtenForm ?Fg .
            FILTER (
                LANGMATCHES(LANG(?Fg), "fr")
            )
            OPTIONAL {{
                {query_concept} sct:definition ?Dg .
                FILTER(
                    CONTAINS(STR(?Dg), STR(?Fg)) && LANGMATCHES(LANG(?Dg), "fr")
                )
            }}
        }}
        """
        object_converted = rdfdb.issue_query(query_convert, to_table=True)
        if object_converted.shape[0] > 0:
            if object_converted.shape[0] > 1:
                # print(f"[WARN] There is more than 1 gibberish representation for {object}.")
                object_converted_ = object_converted.dropna(how="any", axis=0)
            else:
                object_converted_ = object_converted
            object_converted = object_converted_["?F"].iloc[0] if object_converted_.shape[0] > 0 else object_converted["?F"].iloc[0]
        else:
            object_converted = object

        triples_real_converted.add(
            (subject_converted, predicate, object_converted)
        )
    return triples_real_converted, triples_fake

def metrics_wrt_real(query_concept, real_answer, fake_answer, rdfdb):
    real_t, fake_t = correspondence_wrt_real(query_concept, real_answer, fake_answer, rdfdb)
    intersection = real_t.intersection(fake_t) #tp
    fns = real_t - intersection # Relevant but not detected
    fps = fake_t - intersection # Irrelevant but detected: false alarm
    return len(intersection), len(fns), len(fps)


### UTILS ###

def get_full_uri(shortened_uri):
    id = shortened_uri.split("id:")[-1]
    return f"<https://en-word.net/id/{id}>"

class Triple(BaseModel):
    subject: str = "The subject (e.g. the subclass)"
    predicate: str = "The predicate (e.g. subClassOf)"
    object: str = "The object (e.g. the superclass)"

class ExtractedOntology(BaseModel):
    thoughts: str = Field(description="Write your justifications here.")
    triples: list[Triple] = [
        Triple(subject="subclass", predicate="subClassOf", object="superclass"),
        Triple(subject="part", predicate="partOf", object="whole")
    ]

class HypernymyAnswer(BaseModel):
    concept_a : Optional[str]
    concept_b : Optional[str]
    full_answer: Optional[str] = Field(description="The full answer.")
    thoughts: Optional[str] = Field(description="Write your justifications here.")
    answer : Optional[bool]

def evaluate_relation_extraction(args):
    # Read results
    path_results = Path(args.results_dir)
    assert path_results.exists(), f"Results directory {args.results_dir} does not exist."
    result_files = list(path_results.glob("*.json"))
    concept_set = set()

    # Get the concepts
    for result_file in result_files:
        # Parse id
        shortened_uri = result_file.stem
        shortened_uri = shortened_uri.split("_")[0]
        full_uri = get_full_uri(shortened_uri)
        concept_set.add(full_uri)

    # Get the results now
    concept_list = list(concept_set)
    answers_real = []
    answers_fake = []

    for concept in concept_list:
        file_name = path_results / f"{shorten_uri(concept)}_real.json"
        with open(file_name, "r") as f:
            answers_real.append(ExtractedOntology(
                **json.load(f)
            ))
        file_name = path_results / f"{shorten_uri(concept)}_fake.json"
        with open(file_name, "r") as f:
            answers_fake.append(ExtractedOntology(
                **json.load(f)
            ))
    
    rdfdb = RDFDatabase(
        kg_name=args.kg_name,
        init_kg=False
    )

    tps_real = 0
    fps_real = 0
    fns_real = 0
    n_examples_real = 0
    f1s_real = []

    tps_fake = 0
    fps_fake = 0
    fns_fake = 0
    n_examples_fake = 0
    f1s_fake = []

    # Let's get into the evaluation.
    for concept, answer_real, answer_fake in tqdm(zip(concept_list, answers_real, answers_fake), total=len(concept_list)):
        try:
            tp_set, fp_set, fn_set = get_metrics(concept, answer_real, closure=True, rdfdb=rdfdb)
        except:
            tp_set, fp_set, fn_set = get_metrics(concept, answer_real, closure=False, rdfdb=rdfdb)
        tp = len(tp_set)
        fp = len(fp_set)
        fn = len(fn_set)

        tps_real += tp
        fps_real += fp
        fns_real += fn
        n_examples_real += 1
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        f1s_real.append(2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0)

        try:
            tp_set, fp_set, fn_set = get_metrics(concept, answer_fake, closure=True, lang="fr", rdfdb=rdfdb)
        except:
            tp_set, fp_set, fn_set = get_metrics(concept, answer_fake, closure=False, lang="fr", rdfdb=rdfdb)
        tp = len(tp_set)
        fp = len(fp_set)
        fn = len(fn_set)

        tps_fake += tp
        fps_fake += fp
        fns_fake += fn
        n_examples_fake += 1
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        f1s_fake.append(2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0)

    # Get all scores
    precision_real = tps_real / (tps_real + fps_real) if tps_real + fps_real > 0 else 0
    recall_real = tps_real / (tps_real + fns_real) if tps_real + fns_real > 0 else 0
    f1_real = 2 * (precision_real * recall_real) / (precision_real + recall_real) if precision_real + recall_real > 0 else 0

    precision_fake = tps_fake / (tps_fake + fps_fake) if tps_fake + fps_fake > 0 else 0
    recall_fake = tps_fake / (tps_fake + fns_fake) if tps_fake + fns_fake > 0 else 0
    f1_fake = 2 * (precision_fake * recall_fake) / (precision_fake + recall_fake) if precision_fake + recall_fake > 0 else 0

    print("[INFO] Lax evaluation (with transitive closures)")
    print("Overall precision (GT vs answers on real corpus): ", np.round(precision_real, 3))
    print("Overall precision (GT vs answers on gibberish corpus): ", np.round(precision_fake, 3))
    print("Overall recall (GT vs answers on real corpus): ", np.round(recall_real, 3))
    print("Overall recall (GT vs answers on gibberish corpus): ", np.round(recall_fake, 3))
    print("Overall F1 (GT vs answers on real corpus): ", np.round(f1_real, 3))
    print("Overall F1 (GT vs answers on gibberish corpus): ", np.round(f1_fake, 3))
    print("Mean individual F1 real: ", np.round(np.mean(f1s_real), 3))
    print("Mean individual F1 gibberish: ", np.round(np.mean(f1s_fake), 3))
    print("\n")
    print("[INFO] Alignment evaluation")

    tp_all = 0
    fn_all = 0
    fp_all = 0

    f1_per_sample = []
    n_samples = 0

    for x, answer_real, answer_fake in tqdm(zip(concept_list, answers_real, answers_fake)):
        # Check that there are real triples
        if len(answer_real.triples) == 0:
            continue

        tp, fn, fp = metrics_wrt_real(x, answer_real, answer_fake, rdfdb)
        n_samples += 1
        # Compute local metrics
        prec = tp / (tp + fp) if (tp + fp > 0) else 0
        rec = tp / (tp + fn) if (tp + fn > 0) else 0
        f1_per_sample.append((2 * (prec * rec) / (prec + rec)) if (prec > 0 or rec > 0) else 0)
        tp_all += tp
        fn_all += fn
        fp_all += fp

    overall_prec = tp_all / (tp_all + fp_all)
    overall_rec = tp_all / (tp_all + fn_all)
    overall_f1 = 2 * (overall_prec * overall_rec) / (overall_prec + overall_rec)

    print("number of real samples with no true positives (skipped):", len(concept_list) - n_samples)
    print("Average F1 per sample :", np.round(np.mean(f1_per_sample), 3))
    print("\n")
    print("Overall alignment precision : ", np.round(overall_prec, 3))
    print("Overall alignment recall : ", np.round(overall_rec, 3))
    print("Overall alignment F1 :", np.round(overall_f1, 3))

def evaluate_taxonomy_discovery(args):
    path_results = Path(args.results_dir)
    assert path_results.exists(), f"Results directory {args.results_dir} does not exist."
    try:
        positives_dir = path_results / "positives"
        negatives_dir = path_results / "negatives"

        assert positives_dir.exists(), f"Positives directory {positives_dir} does not exist."
        assert negatives_dir.exists(), f"Negatives directory {negatives_dir} does not exist."
    except:
        positives_dir = path_results / "positive"
        negatives_dir = path_results / "negative"

        assert positives_dir.exists(), f"Positives directory {positives_dir} does not exist."
        assert negatives_dir.exists(), f"Negatives directory {negatives_dir} does not exist."

    # Get all file names
    positives_files = list(positives_dir.glob("*.json"))
    negatives_files = list(negatives_dir.glob("*.json"))
    positive_stems = set([x.stem for x in positives_files])
    negative_stems = set([x.stem for x in negatives_files])

    # Get all numbers
    idx_positives = set(
        [x.stem.split("_")[0] for x in positives_files]
    )
    idx_negatives = set(
        [x.stem.split("_")[0] for x in negatives_files]
    )

    # Assert that for each positive, there is both a real and a gibberish version
    for idx in idx_positives:
        assert f"{idx}_real" in positive_stems, f"Missing real file for {idx}."
        assert f"{idx}_fake" in positive_stems, f"Missing fake file for {idx}."
    
    # Assert that for each negative, there is both a real and a gibberish version
    for idx in idx_negatives:
        assert f"{idx}_real" in negative_stems, f"Missing real file for {idx}."
        assert f"{idx}_fake" in negative_stems, f"Missing fake file for {idx}."

    # Read all files
    answers_true_real = []
    answers_true_fake = []
    for i in tqdm(idx_positives):
        file_name = positives_dir / f"{i}_real.json"
        if file_name.exists():
            with open(file_name, "r") as f:
                answers_true_real.append(HypernymyAnswer(
                    **json.load(f)
                ))
        else:
            print(f"[ERROR] {file_name} does not exist")
        
        file_name = positives_dir / f"{i}_fake.json"
        if file_name.exists():
            with open(file_name, "r") as f:
                answers_true_fake.append(HypernymyAnswer(
                    **json.load(f)
                ))
        else:
            print(f"[ERROR] {file_name} does not exist")

    answers_false_real = []
    answers_false_fake = []
    for i in tqdm(idx_negatives):
        file_name = negatives_dir / f"{i}_real.json"
        if file_name.exists():
            with open(file_name, "r") as f:
                answers_false_real.append(HypernymyAnswer(
                    **json.load(f)
                ))
        else:
            print(f"[ERROR] {file_name} does not exist")
        file_name = negatives_dir / f"{i}_fake.json"
        if file_name.exists():
            with open(file_name, "r") as f:
                answers_false_fake.append(HypernymyAnswer(
                    **json.load(f)
                ))
        else:
            print(f"[ERROR] {file_name} does not exist")

    map_answers_clf = {
        True: 1,
        False:0,
        None:-1
    }

    answers_true_real_vector = np.array([map_answers_clf[a.answer] for a in answers_true_real])
    answers_true_fake_vector = np.array([map_answers_clf[a.answer] for a in answers_true_fake])
    answers_false_real_vector = np.array([map_answers_clf[a.answer] for a in answers_false_real])
    answers_false_fake_vector = np.array([map_answers_clf[a.answer] for a in answers_false_fake])
    ground_truth = np.array(([1] * answers_true_real_vector.shape[0]) + ([0] * answers_false_real_vector.shape[0]))

    answers_real_vector = np.concatenate(
        (answers_true_real_vector, answers_false_real_vector)
    )
    answers_fake_vector = np.concatenate(
        (answers_true_fake_vector, answers_false_fake_vector)
    )

    # Classification report for the real case
    print("[INFO] Real dataset vs ground-truth")

    print(classification_report(
        ground_truth,
        answers_real_vector,
        digits=3
    ))

    print("[INFO] Gibberish dataset vs ground-truth")

    print(classification_report(
        ground_truth,
        answers_fake_vector,
        digits=3
    ))

    precision_real = precision_score(ground_truth, answers_real_vector, pos_label=1, average=None)
    precision_fake = precision_score(ground_truth, answers_fake_vector, pos_label=1, average=None)

    recall_real = recall_score(ground_truth, answers_real_vector, pos_label=1, average=None)
    recall_fake = recall_score(ground_truth, answers_fake_vector, pos_label=1, average=None)

    f1_real = f1_score(ground_truth, answers_real_vector, pos_label=1, average=None)
    f1_fake = f1_score(ground_truth, answers_fake_vector, pos_label=1, average=None)

    # vstack all the vectors
    metrics_all = np.vstack([
        precision_real[-2:],
        recall_real[-2:],
        f1_real[-2:],
        precision_fake[-2:],
        recall_fake[-2:],
        f1_fake[-2:]
    ])
    metrics_all = (metrics_all.sum(axis=1) / 2).round(3)

    print("[INFO] Macro-averaged metrics")
    print("Precision real: ", metrics_all[0])
    print("Recall real: ", metrics_all[1])
    print("F1 real: ", metrics_all[2])
    print("Precision fake: ", metrics_all[3])
    print("Recall fake: ", metrics_all[4])
    print("F1 fake: ", metrics_all[5])

    # Prediction alignment
    mask = answers_real_vector >= 0
    print("[INFO] Real dataset vs Gibberish dataset")
    print(classification_report(
        answers_real_vector[mask],
        answers_fake_vector[mask],
        digits=3
    ))
    # Compute macro-averaged metrics on 0 and 1
    precision_real = precision_score(answers_real_vector[mask], answers_fake_vector[mask], pos_label=1, average=None)
    recall_real = recall_score(answers_real_vector[mask], answers_fake_vector[mask], pos_label=1, average=None)
    f1_real = f1_score(answers_real_vector[mask], answers_fake_vector[mask], pos_label=1, average=None)

    metrics_all = np.vstack([
        precision_real[-2:],
        recall_real[-2:],
        f1_real[-2:]
    ])
    metrics_all = (metrics_all.sum(axis=1) / 2).round(3)
    print("[INFO] Macro-averaged alignment metrics")
    print("Precision: ", metrics_all[0])
    print("Recall: ", metrics_all[1])
    print("F1: ", metrics_all[2])



def main(args):
    task = args.task
    assert task in ["relation_extraction", "taxonomy_discovery"]
    if task == "relation_extraction":
        evaluate_relation_extraction(args)
    else:
        evaluate_taxonomy_discovery(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Task to evaluate.")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory.")
    parser.add_argument("--kg_name", type=str, required=False, help="Knowledge graph name.")    

    args = parser.parse_args()
    # If the task is relation extraction, we need kg_name
    # Moreover make sure that the endpoint is running and the KG is initialized.
    if args.task == "relation_extraction":
        assert args.kg_name, "Knowledge graph name is required for relation extraction (e.g. WordNet-sweets)."
    main(args)