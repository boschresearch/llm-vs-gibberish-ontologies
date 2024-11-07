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

# Transformer LLMs
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from pathlib import Path
from typing import Union

def load_base_model(model_name="meta-llama/Llama-2-7b-hf",
                    load_in_8bit : bool =False,
                    load_in_4bit : bool =True,
                    directory : Union[str, Path] = ".",
                    model_kwargs : dict = {}
                    ):
    
    assert (load_in_8bit and not load_in_4bit) or (load_in_4bit and not load_in_8bit)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    device_map = "auto"
    torch_dtype = torch.bfloat16

    if "flan" in model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map, trust_remote_code=True,
            torch_dtype=torch_dtype,
            cache_dir = directory,
            **model_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map, trust_remote_code=True,
            torch_dtype=torch_dtype,
            cache_dir = directory,
            **model_kwargs
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir = directory
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


