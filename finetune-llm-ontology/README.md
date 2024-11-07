# finetune-llm-ontology

Code of paper: Do LLMs Really Adapt to Domains? An Ontology Learning Perspective  (ISWC 2024)

## Description

This project is meant for fine-tuning open-source LLMs on WordNet ontologies, or evaluating them for relation extraction and taxonomy discovery.

## Dependencies

In addition to the dependencies in the `requirements.txt` file, this project depends on the `wordnet-gibberish` package, which can be found [here](https://anonymous.4open.science/r/wordnet-gibberish-9D17/README.md).

To be accurate, the datasets created by this project are required for the project to fine-tune or evaluate LLMs on WordNet ontologies.

To setup the project, run the following command:

```bash
pip instal -e <path/to/wordnet-gibberish>
pip install -r requirements.txt
pip install -e .
```

## Datasets

The datasets used for the experiments can be found [here](https://figshare.com/s/7b9b51d²f1eb52fc9da42). The directory `fake` contains the WordNet-based Knowledge Graphs (`.ttl` files) and the rules (`.dlog` files, usable by RDFox) created by the `wordnet-gibberish` package, required for evaluation. The `extraction_datasets` directory contains the datasets used for the relation extraction experiments, and the `hypernymy_datasets` directory contains the datasets used for the taxonomy discovery experiments.

## Prompting for off-the-shelf evaluation

### Models

In our paper, we evaluate the following models on relation extraction and taxonomy discovery tasks:
- GPT-3.5
- GPT-4
- meta-llama/Llama-2-13b-chat-hf
- HuggingFaceH4/zephyr-7b-beta

### Prompts

The table below sums up the characteristics of the prompts used for the experiments in the paper:

#### Relation extraction Prompts

| Model | CoT | Zero-shot | One-shot | Few-shot |
| --- | --- | --- | --- | --- |
| GPT-3.5 | :white_check_mark: |   | :white_check_mark: |   |
| GPT-4 | :white_check_mark: |   | :white_check_mark: |   |
| LLaMa2-13B | :white_check_mark: |   | :white_check_mark: |   |
| Zephyr-7B-beta | :white_check_mark: |   | :white_check_mark: |   |

#### Taxonomy discovery Prompts

| Model | CoT | Zero-shot | One-shot | Few-shot |
| --- | --- | --- | --- | --- |
| GPT-3.5 | :white_check_mark: |   |   | :white_check_mark: |
| GPT-4 |   |   | :white_check_mark: |   |
| LLaMa2-13B | :white_check_mark: |   | :white_check_mark: |   |
| Zephyr-7B-beta | :white_check_mark: |   | :white_check_mark: |   |


### Prompting HuggingFace LLMs

To prompt an off-the-shelf LLM from HuggingFace on a WordNet ontology for relation extraction, run the following command:

```bash
python relation_extraction_main.py --dataset_path <dataset_path> --save_path <save_path> --model_name <model_name> --cache_path <huggingface_cache_path>
```

To prompt an off-the-shelf LLM on a WordNet ontology for taxonomy discovery, run the following command:

```bash
python taxonomy_discovery_main.py --positive_path <path/to/positive/dataset> --negative_path <path/to/negative/dataset> --save_path <save_path> --model_name <model_name> --cache_path <huggingface_cache_path>
```

### Prompting for OpenAI models

In our experiments, we used Azure's API to prompt OpenAI's models. The following libraries are required:
```bash
pip install python-dotenv
pip install -U langchain
pip install langchain-openai
```

To prompt OpenAI's models on a WordNet ontology for relation extraction, run the following command:

```bash
python relation_extraction_gpt.py --model <model_name> --extraction_dataset <extraction_dataset> --save_dir <save_dir>
```

To prompt OpenAI's models on a WordNet ontology for taxonomy discovery, run the following command:

```bash
python taxonomy_discovery_gpt.py --model <model_name> --positives <path/to/positive/dataset> --negatives <path/to/negative/dataset> --save_dir <save_dir> --cot --shots <shots>
```
where `shots` is either `zero`, `one`, or `few`, and the flag `--cot` is used to indicate that the model should be prompted with the CoT prompt (remove it if you want to prompt the model without the CoT prompt).

### Evaluation

For the evaluation of LLMs on OL tasks, the `swiplserver` package is required. To install it, follow the instructions [here](https://www.swi-prolog.org/packages/mqi/prologmqi.html).

To evaluate a LLM on relation extraction, run the following command:

```bash
python ots_evaluate.py --task relation_extraction --results_dir <path/to/results/directory> --kg_name <kg_name>
```
For this evaluation, a RDFox datastore containing the gibberish dataset (created by `wordnet-gibberish`) is required. The `kg_name` argument is used to specify the name of the RDFox datastore (e.g., `WordNet-sweets`).

To evaluate a LLM on taxonomy discovery, run the following command:
```bash
python ots_evaluate.py --task taxonomy_discovery --results_dir <path/to/results/directory>
```
As the positives and negatives are already separated by the prompting scripts, it is not necessary to have a RDFox datastore for this evaluation.

## Fine-tuning

To fine-tune a LLM on a WordNet ontology for taxonomy discovery, run the following command:

```bash
python main.py --positive_path <path/to/positive/dataset> --negative_path <path/to/negative/dataset> --model_name <model_name> --output_dir <output_dir> --cache_path <huggingface_cache_path> --evaluate --num_train_epochs 20 --learning_rate 3e-6 --lora_r 1024 --lora_alpha 256 --gibberish
```
The data is split (pseudo-randomly) into training and testing sets. The model is trained on the training set and evaluated on the testing set. The model is saved in the `output_dir` directory. The --gibberish flag is used to indicate that the model should be fine-tuned on the gibberish dataset or not.

## Transfer experiment

Using a fine-tuned LLM, you can test its performance on a different WordNet ontology for taxonomy discovery. To do so, run the following command:

```bash
python domain_transfer.py --positive_path <path/to/positive/dataset> --negative_path <path/to/negative/dataset> --model_path <model_path> --output_dir <output_dir> --gibberish
```
Using the same split as the fine-tuning experiment, the model is evaluated on the testing set. The metrics are displayed, and the results are saved in the `output_dir` directory.

## Purpose of this Software

The software is a research prototype, solely developed and published as supporting material for the research paper cited below. It will not be maintained or monitored in any way.

## License

This software is open-sourced under the AGPL-3.0 license. See the `LICENSE` file for details.

The Open English WordNet (2023 Edition) is released under the Creative Commons Attribution 4.0 International License. See their LICENSE file [here](https://github.com/globalwordnet/english-wordnet/blob/main/LICENSE.md) for details.

## Citation

If you use our software or datasets generated by it, please cite our [paper](https://arxiv.org/abs/2407.19998):

```
@misc{mai2024llmsreallyadaptdomains,
      title={Do LLMs Really Adapt to Domains? An Ontology Learning Perspective}, 
      author={Huu Tan Mai and Cuong Xuan Chu and Heiko Paulheim},
      year={2024},
      eprint={2407.19998},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.19998}, 
}
```

## Contact

For any inquiries or questions, please contact the project maintainer at [huutan.mai@de.bosch.com](mailto:huutan.mai@de.bosch.com).
