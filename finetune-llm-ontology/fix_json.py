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

from finetune_ontology.relation_extraction import ExtractedOntology, parse_answer_json, parse_answer_json_2
from finetune_ontology.taxonomy_discovery import HypernymyAnswer
from tqdm import tqdm
import argparse
import json
import jsonstrip
from pathlib import Path

def main(args):
    task = args.task
    if task == "relation_extraction":
        print("[INFO] Fixing relation extraction outputs")
    elif task == "taxonomy_discovery":
        print("[INFO] Fixing taxonomy discovery outputs")

    if task == "relation_extraction":
    
        results_dir = Path(args.results_dir)
        results_files = list(results_dir.glob("*.json"))

        # Count how many files are invalid
        invalid_files = 0
        invalid_files_list = []
        for results_file in results_files:
            with open(results_file, "r") as f:
                results = json.load(f)
            
            thoughts = results.get("thoughts", None)
            if not thoughts or (isinstance(thoughts, str) and "Could not parse the result" in thoughts):
                invalid_files += 1
                invalid_files_list.append(
                    Path(results_file)
                )
        print("[INFO] Number of invalid files: ", invalid_files)
        fixed = 0
        for results_file in tqdm(invalid_files_list):
            with open(results_file, "r") as f:
                results = json.load(f)
            
            thoughts = results.get("thoughts", None)
            if not thoughts or (isinstance(thoughts, str) and "Could not parse the result" in thoughts):
                print(f"[INFO] Fixing {results_file}")
                full_answer = results["full_answer"].strip()

                try:
                    # Parse json again
                    json_object = parse_answer_json(full_answer)
                    if json_object is None:
                        json_object = parse_answer_json_2(full_answer)
                    results_new = ExtractedOntology(
                        full_answer = full_answer,
                        **json.loads(jsonstrip.strip(json_object))
                    )
                    # Write
                    try:
                        json_string = results_new.json(indent=4)
                    except:
                        json_string = results_new.model_dump_json(indent=4)
                    with open(results_file, "w") as f:
                        f.write(json_string)
                    fixed += 1
                except Exception as e:
                    print(e)
                    print(f"[ERROR] Could not parse the result from :\n{full_answer}")
    
    elif task == "taxonomy_discovery":
        results_dir = Path(args.results_dir)
        results_files = list(results_dir.glob("*.json"))

        # Count how many files are invalid
        invalid_files = 0
        invalid_files_list = []
        for results_file in results_files:
            with open(results_file, "r") as f:
                results = json.load(f)
            
            thoughts = results.get("thoughts", None)
            if not thoughts or (isinstance(thoughts, str) and "Could not parse the result" in thoughts):
                invalid_files += 1
                invalid_files_list.append(
                    Path(results_file)
                )
        print("[INFO] Number of invalid files: ", invalid_files)
        fixed = 0
        for results_file in tqdm(invalid_files_list):
            with open(results_file, "r") as f:
                results = json.load(f)
            
            thoughts = results.get("thoughts", None)
            if not thoughts or (isinstance(thoughts, str) and "Could not parse the result" in thoughts):
                print(f"[INFO] Fixing {results_file}")
                full_answer = results["full_answer"].strip()

                try:
                    # Parse json again
                    json_object = parse_answer_json(full_answer)
                    if json_object is None:
                        json_object = parse_answer_json_2(full_answer)
                    # if that does not work either, we will try to parse the first occurrence of "answer": xxx
                    if json_object is None:
                        idx_answer = full_answer.find("answer\": ")
                        shift = len("answer\": ")
                        if idx_answer == -1:
                            idx_answer = full_answer.find("answer\" : ")
                            shift = len("answer\" : ")
                        
                        if idx_answer != -1:
                            answer_ = full_answer[idx_answer + shift:]
                            if answer_.startswith("true") and not(answer_.startswith("true or false")):
                                answer_ = True
                            elif answer_.startswith("false"):
                                answer_ = False
                            else:
                                answer_ = None
                                raise Exception("Could not parse the answer with first occurence")
                        
                        results_new = HypernymyAnswer(
                            concept_a=results["concept_a"],
                            concept_b=results["concept_b"],
                            full_answer = full_answer,
                            thoughts = "Parsed using first occurrence.",
                            answer = answer_
                        )
                    else:
                        results_new = HypernymyAnswer(
                            concept_a=results["concept_a"],
                            concept_b=results["concept_b"],
                            full_answer = full_answer,
                            **json.loads(jsonstrip.strip(json_object))
                        )
                    # Write
                    try:
                        json_string = results_new.json(indent=4)
                    except:
                        json_string = results_new.model_dump_json(indent=4)
                    with open(results_file, "w") as f:
                        f.write(json_string)
                    fixed += 1
                except Exception as e:
                    print(e)
                    print(f"[ERROR] Could not parse the result from :\n{full_answer}")
    
    print(f"[INFO] Fixed {fixed} out of {invalid_files}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix JSON files")
    parser.add_argument("--task", type=str, required=True, help="Task to fix")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing the results")
    args = parser.parse_args()
    main(args)
