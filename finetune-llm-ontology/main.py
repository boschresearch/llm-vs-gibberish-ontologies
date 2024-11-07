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
from finetune_ontology.train import make_trainer, LORA_CONFIG_DEFAULT
from finetune_ontology.llm import load_base_model
from sklearn.metrics import classification_report, f1_score
from peft import LoraConfig
from pathlib import Path
import pandas as pd
import argparse


def main(args):
    print(args)

    positive_path = Path(args.positive_path)
    negative_path = Path(args.negative_path)
    model, tokenizer = load_base_model(
        model_name=args.model_name if args.model_name else "meta-llama/Llama-2-7b-hf",
        directory=args.cache_path
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # LoRA config
    if args.lora_r or args.lora_alpha:
        lora_config = LoraConfig(
            r=args.lora_r if args.lora_r else 64,
            lora_alpha=args.lora_alpha if args.lora_alpha else 16,
            bias="none",
            lora_dropout=0.05,  # Conventional
            task_type="CAUSAL_LM",
        )
    else:
        lora_config = LORA_CONFIG_DEFAULT

    if args.validate:
        train_datasets, val_datasets, eval_dataset = make_hypernym_datasets(
            positive_path, negative_path, args.gibberish, validation=True, folds=args.folds, seed_val=args.seed_val
        )
        f1_scores = []
        for i, (train_dataset, val_dataset) in enumerate(zip(train_datasets, val_datasets)):
            trainer = make_trainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                output_dir=output_dir,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.learning_rate,
                logging_steps=args.logging_steps,
                num_train_epochs=args.num_train_epochs,
                max_steps=args.max_steps,
                log_with=args.log_with,
                save_steps=args.save_steps,
                save_total_limit=args.save_total_limit,
                push_to_hub=args.push_to_hub,
                hub_model_id=args.hub_model_id,
                lora_config=lora_config
            )

            trainer.train()
            # Evaluate on validation
            print(f"[INFO] Evaluating on Validation split {i}")
            queries, results, labels = evaluate(model, tokenizer, val_dataset)
            print(classification_report(labels, results, digits=3))
            # Save results
            df = pd.DataFrame({"queries": queries, "results": results, "labels": labels})
            df.to_csv(
                output_dir / f"results_{'fake' if args.gibberish else 'real'}_after_val_{i}.csv", index=False
            )
            f1_scores.append(f1_score(labels, results, average="macro"))
            print("[INFO] Evaluating on Test split")
            queries, results, labels = evaluate(model, tokenizer, eval_dataset)
            print(classification_report(labels, results, digits=3))
            # Save results
            df = pd.DataFrame({"queries": queries, "results": results, "labels": labels})
            df.to_csv(
                output_dir / f"results_{'fake' if args.gibberish else 'real'}_after_test_{i}.csv", index=False
            )

        print(f"[INFO] Validation F1 Scores: {f1_scores}")
        print(f"[INFO] Mean Validation F1 Score: {sum(f1_scores) / len(f1_scores)}")

    else:
        train_dataset, eval_dataset = make_hypernym_datasets(
            positive_path, negative_path, args.gibberish
        )

        trainer = make_trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            log_with=args.log_with,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
            lora_config=lora_config
        )

        trainer.save_model(output_dir)

        # Evaluate on train dataset
        print("[INFO] Evaluating on Train split")
        queries, results, labels = evaluate(model, tokenizer, train_dataset)
        print(classification_report(labels, results, digits=3))
        # Save results
        df = pd.DataFrame({"queries": queries, "results": results, "labels": labels})
        df.to_csv(
            output_dir / f"results_{'fake' if args.gibberish else 'real'}_before_train.csv", index=False
        )
        # Evaluation before
        print("[INFO] Evaluating on Test split")
        queries, results, labels = evaluate(model, tokenizer, eval_dataset)
        print(classification_report(labels, results, digits=3))
        # Save results
        df = pd.DataFrame({"queries": queries, "results": results, "labels": labels})
        df.to_csv(
            output_dir / f"results_{'fake' if args.gibberish else 'real'}_before_test.csv", index=False
        )

        trainer.train()
    
        if args.evaluate:
            queries, results, labels = evaluate(model, tokenizer, train_dataset)
            print("[INFO] Final Evaluation on Train split")
            print(classification_report(labels, results, digits=3))
            # Save results
            df = pd.DataFrame({"queries": queries, "results": results, "labels": labels})
            df.to_csv(
                output_dir / f"results_{'fake' if args.gibberish else 'real'}_after_train.csv", index=False
            )
            print("[INFO] Final Evaluation on Test split")
            queries, results, labels = evaluate(model, tokenizer, eval_dataset)
            print(classification_report(labels, results, digits=3))
            # Save results
            df = pd.DataFrame({"queries": queries, "results": results, "labels": labels})
            df.to_csv(
                output_dir / f"results_{'fake' if args.gibberish else 'real'}_after_test.csv", index=False
            )
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive_path", type=str, required=True)
    parser.add_argument("--negative_path", type=str, required=True)
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--gibberish", action="store_true")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--log_with", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=10)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed_val", type=int, default=42)
    args = parser.parse_args()    

    main(args)