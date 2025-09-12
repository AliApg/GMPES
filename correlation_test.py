import itertools
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


def _filter_by_name(
    pattern: str, file_list: List[Union[str, Path]]
) -> List[Union[str, Path]]:
    """Filter a list of file paths by a pattern in their base names."""

    return [file for file in file_list if pattern in os.path.basename(file)]


def _load_pickle_files(
    file_paths: List[Union[str, Path]],
) -> Dict[str, np.ndarray]:
    """Load and concatenate data from multiple pickle files."""

    sentences_a_embeddings = []
    sentences_b_embeddings = []

    for file_path in file_paths:
        with open(file_path, "rb") as _f:
            data = pickle.load(_f)
        sentences_a_embeddings.append(data["sentences_a_embeddings"])
        sentences_b_embeddings.append(data["sentences_b_embeddings"])

    return {
        "sentences_a_embeddings": np.concatenate(sentences_a_embeddings),
        "sentences_b_embeddings": np.concatenate(sentences_b_embeddings),
    }


def _train_test_splitting(models) -> Dict:
    """
    Splits and organizes model data by loading train and test pickle files for each model.

    This function processes a list of model paths and their corresponding names, loading
    associated pickle files. If train files are available, it combines them with test files;
    otherwise, it loads all provided files for the model. The data is organized into a
    dictionary where each key is a model name, and the value is the loaded data.

    Args:
        models (List[Tuple[List[str], str]]): A list of tuples, where each tuple contains
            a list of file paths for a model and the model's name.

    Returns:
        Dict: A dictionary mapping model names to their loaded data from pickle files.
    """

    result = {}
    for model_paths, model_name in models:
        _data = {}
        if train_files := _filter_by_name("train", model_paths):  # type: ignore
            _data = _load_pickle_files(
                train_files + _filter_by_name("test", model_paths)  # type: ignore
            )

        else:
            _data = _load_pickle_files(model_paths)

        result[model_name] = _data
    return result


def pair_cos_sim_name(embeddings, models: list, index_1: int, index_2: int):
    """
    Computes the average cosine similarity between sentence embeddings of two models.

    This function calculates the cosine similarity between pairs of sentence embeddings
    (sentences_a and sentences_b) from two specified models, identified by their indices
    in the models list. The embeddings are concatenated, and the cosine similarity is
    computed for each pair, with the average similarity returned.

    Args:
        embeddings (Dict): A dictionary containing sentence embeddings for each model,
            with keys as model names and values including 'sentences_a_embeddings' and
            'sentences_b_embeddings'.
        models (List[str]): A list of model names.
        index_1 (int): Index of the first model in the models list.
        index_2 (int): Index of the second model in the models list.

    Returns:
        float: The average cosine similarity between the sentence embeddings of the two models.
    """

    cosine_sim = []

    for ab_model_1, ab_model_2 in zip(
        np.concatenate(
            [
                embeddings[models[index_1]]["sentences_a_embeddings"],
                embeddings[models[index_1]]["sentences_b_embeddings"],
            ]
        ),
        np.concatenate(
            [
                embeddings[models[index_2]]["sentences_a_embeddings"],
                embeddings[models[index_2]]["sentences_b_embeddings"],
            ]
        ),
    ):
        cosine_sim.append(
            cosine_similarity(
                ab_model_1.reshape(1, -1),
                ab_model_2.reshape(1, -1),
            )[0][0]
        )

    return float(np.mean(cosine_sim))


def generate_combinations(
    start: int,
    end: int,
    combination_length: int,
) -> list:
    """
    Generates all possible combinations of indices within a specified range.

    This function uses itertools.combinations to create a list of all possible combinations
    of indices from 'start' to 'end' (inclusive) with the specified combination length.

    Args:
        start (int): The starting index of the range (inclusive).
        end (int): The ending index of the range (inclusive).
        combination_length (int): The length of each combination.

    Returns:
        list: A list of tuples, where each tuple represents a combination of indices.
    """
    return list(
        itertools.combinations(range(start, end + 1), combination_length)
    )


def compute_model_pair_cosine_similarities():
    """
    Computes cosine similarities between pairs of model embeddings across multiple datasets.

    This function defines dictionaries of models with 768 and 1024 dimensional embeddings,
    combines them, and generates all possible pairs of models. For each pair and dataset
    (e.g., SICK-R, STS12-16, STSB), it loads the relevant embeddings using
    `_train_test_splitting`, calculates the average cosine similarity between sentence
    embeddings using `pair_cos_sim_name`, and stores the results in a dictionary. The
    results are saved to a JSON file named 'models_corr.json'. If an error occurs during
    similarity computation for a model pair, it is logged and skipped.

    Returns:
        None: The function saves the results to 'models_corr.json' and does not return a value.

    Notes:
        - Models are loaded from predefined paths under 'Embeddings/Size 768' and 'Embeddings/Size 1024'.
        - Datasets processed include 'sickr', 'sts12', 'sts13', 'sts14', 'sts15', 'sts16', and 'stsb'.
        - The function uses `tqdm` for progress tracking of model combinations and datasets.
        - Errors during cosine similarity computation are caught and printed without stopping execution.
    """

    models_768 = {
        "Contriever": sys.path[0] + r"\Embeddings\Size 768\Contriever\Base",
        "Contriever msmarco": sys.path[0]
        + r"\Embeddings\Size 768\Contriever\msmarco",
        "DiffCSE BERT base uncased": sys.path[0]
        + r"\Embeddings\Size 768\DiffCSE\Bert base uncased",
        "DiffCSE RoBERTa base": sys.path[0]
        + r"\Embeddings\Size 768\DiffCSE\RoBERTa base",
        "E5 base": sys.path[0] + r"\Embeddings\Size 768\E5",
        "GTE base": sys.path[0] + r"\Embeddings\Size 768\GTE",
        "InfoCSE bert base": sys.path[0] + r"\Embeddings\Size 768\InfoCSE",
        "SBERT WK bert base uncased": sys.path[0]
        + r"\Embeddings\Size 768\SBERT-WK",
        "WhiteningBERT bert base cased": sys.path[0]
        + r"\Embeddings\Size 768\WhiteningBERT",
        "SimCSE RoBERTa base": sys.path[0] + r"\Embeddings\Size 768\SimCSE",
        "T5 3b": sys.path[0] + r"\Embeddings\Size 768\T5\3b single",
        "T5 11b": sys.path[0] + r"\Embeddings\Size 768\T5\11b single",
        "T5 base": sys.path[0] + r"\Embeddings\Size 768\T5\Base",
        "T5 large": sys.path[0] + r"\Embeddings\Size 768\T5\Large",
    }
    models_1024 = {
        "BGE m3": sys.path[0] + r"\Embeddings\Size 1024\BGE",
        "GTE large": sys.path[0] + r"\Embeddings\Size 1024\GTE",
        "InfoCSE bert large": sys.path[0] + r"\Embeddings\Size 1024\InfoCSE",
        "SimCSE RoBERTa large": sys.path[0] + r"\Embeddings\Size 1024\SimCSE",
        "T5 3b (HF)": sys.path[0] + r"\Embeddings\Size 1024\T5\3b single",
    }

    all_models = {**models_768, **models_1024}

    all_combo = {}
    all_models_names = list(all_models.keys())

    for i, j in tqdm(
        generate_combinations(0, len(all_models) - 1, 2),
        desc="Model combinations",
    ):
        _models = [
            all_models_names[i],
            all_models_names[j],
        ]

        for ds in tqdm(
            [
                "sickr",
                "sts12",
                "sts13",
                "sts14",
                "sts15",
                "sts16",
                "stsb",
            ],
            desc="Datasets",
            leave=False,
        ):
            embeddings = _train_test_splitting(
                [
                    [
                        [
                            os.path.join(all_models[model], file)
                            for file in _filter_by_name(
                                ds,
                                os.listdir(all_models[model]),  # type: ignore
                            )
                        ],
                        model,
                    ]
                    for model in _models
                ]
            )

            name = f"{_models[0]} | {_models[1]}"

            try:
                sim_acc_2 = pair_cos_sim_name(embeddings, _models, 0, 1)
            except ValueError as e:
                print(name, ":", str(e).replace("\n", " ")[:70])
                continue

            all_combo[name] = {**{ds: sim_acc_2}, **all_combo.get(name, {})}

    with open(sys.path[0] + r"\models_corr.json", "w", encoding="utf-8") as _f:
        json.dump(all_combo, _f)


def main(json_file: str):
    """
    Adds correlation data to model combination results and saves to a new JSON file.

    This function reads a JSON file containing model combination results and a separate
    JSON file with model correlation data ('models correlations.json'). For each model
    combination (containing 2 or 3 models) in the input JSON, it retrieves the
    corresponding cosine similarity correlations for model pairs using the `get_corr`
    helper function and adds them to the combination's data under a 'correlations' key.
    Combinations with a single model or existing correlations are skipped. The updated
    data is saved to a new JSON file with ' with correlation' appended to the original
    filename.

    Args:
        json_file (str): The name of the input JSON file (located in the 'results' directory)
            containing model combination results.

    Returns:
        None: The function saves the updated results to a new JSON file in the 'Results'
            directory and does not return a value.

    Notes:
        - The function expects 'models correlations.json' in the 'Results' directory.
        - Correlations are added for pairs of models in combinations of 2 or 3 models.
        - The `tqdm` library is used to display progress for processing model combinations.
        - The output JSON file is encoded in UTF-8 to support special characters.
    """

    with open(
        os.path.join(sys.path[0], "Results", json_file),
        "r",
        encoding="utf8",
    ) as f:
        json_data = json.load(f)

    with open(
        os.path.join(sys.path[0], "Results", "models correlations.json"),
        "r",
        encoding="utf8",
    ) as f:
        corr_json_data = json.load(f)

    def get_corr(model_1: str, model_2: str):
        _corrs = corr_json_data.get(
            f"{model_1} | {model_2}"
        ) or corr_json_data.get(f"{model_2} | {model_1}")
        return _corrs

    _final = []
    for batch in tqdm(json_data, desc="Model combinations"):
        if len(
            _models := list(batch["model_combination"].values())
        ) == 1 or batch.get("correlations", None):
            continue

        if len(_models) == 2:
            model_1, model_2 = _models

            batch["correlations"] = {
                str(sorted([model_1, model_2])): get_corr(model_1, model_2)
            }

        elif len(_models) == 3:
            model_1, model_2, model_3 = _models

            batch["correlations"] = {
                str(sorted([model_1, model_2])): get_corr(model_1, model_2),
                str(sorted([model_1, model_3])): get_corr(model_1, model_3),
                str(sorted([model_2, model_3])): get_corr(model_2, model_3),
            }

    with open(
        os.path.join(
            sys.path[0],
            "Results",
            f"{json_file.removesuffix('.json')} with correlation.json",
        ),
        "w",
        encoding="utf-8",
    ) as _f:
        json.dump(json_data, _f)


main("results 1.json")
