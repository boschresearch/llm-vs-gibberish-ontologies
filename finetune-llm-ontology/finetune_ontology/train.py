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

from peft import LoraConfig
from datasets import Dataset
from pathlib import Path
from transformers import TrainingArguments
from trl import SFTTrainer
from typing import Union, Optional

LORA_CONFIG_DEFAULT = LoraConfig(
    r=64,
    lora_alpha=16,
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

def make_trainer(model, train_dataset : Dataset, eval_dataset : Dataset,
            output_dir : Optional[Union[str, Path]] = None,
            batch_size : int = 4,
            gradient_accumulation_steps : int = 2,
            learning_rate : float = 1e-5,
            logging_steps : int = 1,
            num_train_epochs : int = 3,
            max_steps : Optional[int] = -1,
            log_with : Optional[str] = None,
            save_steps : Optional[int] = 100,
            save_total_limit : Optional[int] = 10,
            push_to_hub : bool = False,
            hub_model_id : Optional[str] = None,
            lora_config : LoraConfig = LORA_CONFIG_DEFAULT
            ):
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        report_to=log_with,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        max_seq_length=512,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        peft_config=lora_config,
    )

    return trainer

