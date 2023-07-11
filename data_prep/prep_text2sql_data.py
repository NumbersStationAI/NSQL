"""Prepare data for NSText2SQL."""

import hashlib
import json
import multiprocessing
import os
import random
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path

import click
import numpy as np
import yaml
from prompt_formatters import RajkumarFormatter
from rich.console import Console
from text2sql_dataset import (
    CSS2SQL,
    NVBENCH2SQL,
    Criteria2SQL2SQL,
    KummerfeldText2SQL,
    MimicsqlText2SQL,
    SedeText2SQL,
    SpiderText2SQL,
    SqlCreateContext2SQL,
    Squall2SQL,
    Text2SQLData,
    Text2SQLDataset,
    WikiSQL2SQL,
)
from tqdm.auto import tqdm
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

console = Console(soft_wrap=True)


TEXT2SQL_DATA_LOADERS = {
    "kummerfeld": KummerfeldText2SQL,
    "sede": SedeText2SQL,
    "spider": SpiderText2SQL,
    "wikisql": WikiSQL2SQL,
    "mimicsql": MimicsqlText2SQL,
    "criteria2sql": Criteria2SQL2SQL,
    "sql_create_context2sql": SqlCreateContext2SQL,
    "squall": Squall2SQL,
    "css": CSS2SQL,
    "nvbench": NVBENCH2SQL,
}


def process_dataset(
    prompt_formatter: RajkumarFormatter,
    splits: dict[str, list[Text2SQLData]],
    bad_parses: dict[str, int],
    total: dict[str, int],
    text2sql_dataset: Text2SQLDataset,
    hash_experiment_key: str,
) -> None:
    """Process a dataset and add it to the splits."""
    schema = text2sql_dataset.load_schema()
    temp_outfile = f"_temp_text2sql/prep_data/temp_{hash_experiment_key}.jsonl"
    Path(temp_outfile).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(temp_outfile):
        console.print(f"Reading from {temp_outfile}")
        with open(temp_outfile, "r") as in_f:
            loaded_data = json.load(in_f)
    else:
        loaded_data = text2sql_dataset.load_data(schema)
        console.print(f"Saving to {temp_outfile}")
        with open(temp_outfile, "w") as out_f:
            json.dump(loaded_data, out_f)

    formatting_func = partial(
        text2sql_dataset.format_example,
        schema=schema,
        prompt_formatter=prompt_formatter,
    )
    cnt = 0
    for split, split_data in loaded_data.items():
        console.print(f"Found {len(split_data)} examples for {split}.")
        pool = multiprocessing.Pool(
            processes=15,
        )
        for formatted_data in tqdm(
            pool.imap(formatting_func, split_data, chunksize=100),
            total=len(split_data),
            desc=f"Formatting {split}",
        ):
            total[split] += 1
            cnt += 1
            if formatted_data:
                formatted_as_traindata = Text2SQLData(**formatted_data)
                splits[split].append(Text2SQLData(**formatted_data))
                if total[split] <= 20 or cnt <= 20:
                    console.print(f"\n***[yellow]Example {total[split]}[/yellow]***")
                    console.print(
                        json.dumps(
                            formatted_as_traindata.dict(), indent=2, ensure_ascii=False
                        )
                    )
            else:
                bad_parses[split] += 1
        pool.close()
        pool.join()
    console.print(f"Bad parses: {json.dumps(bad_parses, indent=2, ensure_ascii=False)}")


@click.command()
@click.option("--datasets", type=str, required=True, multiple=True)
@click.option(
    "--config_path",
    type=str,
    default=(f"{os.path.join(os.path.dirname(__file__))}/text2sql_data_config.yaml"),
)
@click.option("--output_dir", type=str, default="")
@click.option("--seed", type=int, default=0)
@click.option("--tokenizer_name", type=str, default="Salesforce/codegen-2B-multi")
@click.option("--seq_length", type=int, default=2048)
@click.option("--merge_dev", type=bool, default=False, is_flag=True)
@click.option("--merge_test", type=bool, default=False, is_flag=True)
def build(
    datasets: list[str],
    config_path: str,
    output_dir: str,
    seed: int,
    tokenizer_name: str,
    seq_length: int,
    merge_dev: bool,
    merge_test: bool,
) -> None:
    """Build training data for text2SQL model training.

    Args:
        datasets: the datasets to read - matches on name in config
        config_path: path to config
        output_dir: output directory
        seed: the random seed
        tokenizer_name: the tokenizer to use
        seq_length: max seq_length for training data
        merge_dev: whether merge dev in to train
        merge_test: whether merge test in to train
    """
    to_save_args = locals()
    random.seed(seed)
    np.random.seed(seed)
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    prompt_formatter = RajkumarFormatter()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    config = yaml.safe_load(open(config_path))
    to_save_args["config"] = config

    data_configs = {}
    for data_loader in config["datasets"]:
        data_configs[data_loader["name"]] = data_loader
    assert len(data_configs) == len(
        config["datasets"]
    ), f"Overloaded name in {config_path}"

    datasets = [dataset.lower() for dataset in datasets]
    for dataset in datasets:
        if dataset not in data_configs:
            raise ValueError(f"Dataset {dataset} not supported.")
        if data_configs[dataset]["loader"] not in TEXT2SQL_DATA_LOADERS:
            raise ValueError(f"Loader {data_configs[dataset]['loader']} not supported.")

    data_classes: list[Text2SQLDataset] = [
        TEXT2SQL_DATA_LOADERS[data_configs[dataset]["loader"]](  # type: ignore
            **data_configs[dataset],
            context_length=seq_length,
            tokenizer_name=tokenizer_name,
        )
        for dataset in datasets
    ]

    splits: dict[str, list[Text2SQLData]] = {"train": [], "dev": [], "test": []}
    bad_parses: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)

    for data_class in data_classes:
        console.print(f"[green]Loading[/green] {data_class.name}")
        to_hash_args = to_save_args.copy()
        to_hash_args["dataset_name"] = data_class.name
        hash_experiment_key = hashlib.sha256(
            json.dumps(to_hash_args, sort_keys=True).encode("utf-8")
        ).hexdigest()
        process_dataset(
            prompt_formatter=prompt_formatter,
            splits=splits,
            bad_parses=bad_parses,
            total=total,
            text2sql_dataset=data_class,
            hash_experiment_key=hash_experiment_key,
        )

    if merge_dev:
        splits["train"].extend(splits["dev"])
        splits["dev"] = []
    if merge_test:
        splits["train"].extend(splits["test"])
        splits["test"] = []

    date = datetime.now().strftime("%Y-%m-%d")
    joined_output_dir = Path(output_dir) / date
    joined_output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"Starting length of train: {len(splits['train'])}")

    # Deduplicate training data
    unq_inps = set()
    new_train = []
    for ex in splits["train"]:
        if ex.instruction not in unq_inps:
            new_train.append(ex)
            unq_inps.add(ex.instruction)
    splits["train"] = new_train

    console.print(f"After dedup length of train: {len(splits['train'])}")

    # Get token size statistics
    tokenized_inputs = tokenizer(list(map(lambda x: x.instruction, splits["train"])))
    tokenized_outputs = tokenizer(list(map(lambda x: x.output, splits["train"])))
    input_lengths = [len(x) for x in tokenized_inputs["input_ids"]]
    output_lengths = [len(x) for x in tokenized_outputs["input_ids"]]
    sum_lengths = [x + y for x, y in zip(input_lengths, output_lengths)]

    console.print(
        f"Max input length: {max(input_lengths)}, "
        f"Median length: {np.median(input_lengths)}, "
        f"90th percentile: {np.percentile(input_lengths, 90)}"
    )
    console.print(
        f"Max output length: {max(output_lengths)}, "
        f"Median length: {np.median(output_lengths)}, "
        f"90th percentile: {np.percentile(output_lengths, 90)}"
    )
    console.print(
        f"Percent overflow: {100*sum(x > seq_length for x in sum_lengths)/len(sum_lengths):.2f}"
    )
    console.print(
        f"Max sum length: {max(sum_lengths)}, "
        f"Median length: {np.median(sum_lengths)}, "
        f"85th percentile: {np.percentile(sum_lengths, 85)}"
        f"90th percentile: {np.percentile(sum_lengths, 90)}"
        f"95th percentile: {np.percentile(sum_lengths, 95)}"
    )

    # Save the data
    random.seed(seed)
    random.shuffle(splits["train"])

    for split in splits:
        console.print(
            f"Found {bad_parses[split]} bad parses out of "
            f"{total[split]} ({100*bad_parses[split]/max(total[split], 1): .2f})."
        )
        console.print(
            f"Saving [green]{split} ({len(splits[split])}) "
            f"[/green] data to {joined_output_dir}/{split}.jsonl"
        )
        with open(joined_output_dir / f"{split}.jsonl", "w") as f:
            for formatted_ex in splits[split]:
                f.write(json.dumps(formatted_ex.dict(), ensure_ascii=False) + "\n")
    with open(f"{joined_output_dir}/config.json", "w") as f:
        json.dump(to_save_args, f, indent=4)


if __name__ == "__main__":
    build()
