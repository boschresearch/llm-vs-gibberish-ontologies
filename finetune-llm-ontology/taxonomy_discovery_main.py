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

from finetune_ontology.llm import load_base_model
from finetune_ontology.taxonomy_discovery import pipeline_discovery
from transformers import pipeline
from pathlib import Path
import argparse
import torch

def main(args):
    print(args)

    model_name = args.model_name if args.model_name else "meta-llama/Llama-2-7b-hf"
    if "llama" in model_name.lower():
        from huggingface_hub import login
        from dotenv import load_dotenv
        import os
        load_dotenv()

        login(token=os.environ.get("HF_TOKEN"))

    model, tokenizer = load_base_model(
        model_name=model_name,
        directory=args.cache_path,
        model_kwargs={ "token" : os.environ.get("HF_TOKEN")} if "llama" in model_name.lower() else {}
    )
    model_name = "llama" if "llama" in model_name.lower() else "zephyr"

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # Text generation pipeline
    model.eval()
    with torch.no_grad():
        pipeline_ = pipeline("text-generation", model=model, tokenizer=tokenizer,
                            torch_dtype=model.config.torch_dtype,
                            device_map="auto")
        SAVE_PATH = Path(args.save_path)
        SAVE_PATH.mkdir(exist_ok=True, parents=True)
        stop_words = ["#", "Query", "\nQuery", "Query:"]
        stop_ids = [tokenizer.encode(stop_word)[0] for stop_word in stop_words]
        pipeline_discovery(
            model_name, args.positive_path, args.negative_path, pipeline_,
            eos_token_id=stop_ids,
            save_path=SAVE_PATH,
            verbose=True
        )
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None, help="Model name.")
    parser.add_argument("--cache_path", type=str, default=".", help="Cache path.")
    parser.add_argument("--positive_path", type=str, required=True)
    parser.add_argument("--negative_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True, help="Save path.")


    args = parser.parse_args()

    main(args)