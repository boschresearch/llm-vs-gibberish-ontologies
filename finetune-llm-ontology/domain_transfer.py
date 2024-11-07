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

from finetune_ontology.datasets import make_hypernym_datasets, evaluate
from finetune_ontology.llm import load_base_model
from sklearn.metrics import classification_report
import pandas as pd
from pathlib import Path
import argparse

def main(args):
    # Assert that the output dir exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    assert output_dir.exists(), f"Output directory {output_dir} does not exist."
    assert Path(args.model_path).exists(), f"Model path {args.model_path} does not exist."
    assert Path(args.positive_path).exists(), f"Positive path {args.positive_path} does not exist."
    assert Path(args.negative_path).exists(), f"Negative path {args.negative_path} does not exist."

    _, test_dataset = make_hypernym_datasets(
        positive_path=args.positive_path,
        negative_path=args.negative_path,
        gibberish=args.gibberish,
        validation=False,
    )

    # Load model from path
    model_path = Path(args.model_path)
    model, tokenizer = load_base_model(
        model_name=model_path,
        directory=model_path,
    )
    
    # Evaluation
    print("[INFO] Evaluating on Test split")
    queries, results, labels = evaluate(model, tokenizer, test_dataset)
    print(classification_report(labels, results, digits=3))
    # Save results
    df = pd.DataFrame({"queries": queries, "results": results, "labels": labels})
    df.to_csv(
        output_dir / f"results_{'fake' if args.gibberish else 'real'}_before_test.csv", index=False
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive_path", type=str, required=True)
    parser.add_argument("--negative_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gibberish", action="store_true")
    args = parser.parse_args()
    main(args)