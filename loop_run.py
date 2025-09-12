import argparse
import ctypes
import os
import sys
import time
from contextlib import redirect_stdout
from itertools import product
from multiprocessing import Pool, Value

from tqdm.auto import tqdm

from gmpes import GMPES

# from gmpes import print_results_box


def run_gmpes(params):
    model_path, ds_name, pop_size, num_gen = params
    try:
        with redirect_stdout(open(os.devnull, "w")):  # pylint:disable=W1514
            # pylint:disable=E1123
            gmpes = GMPES(
                model_files_path=(
                    [sys.path[0] + _path for _path in model_path]
                    if isinstance(model_path, list)
                    else [sys.path[0] + model_path]
                ),
                population_size=pop_size,
                num_generations=num_gen,
                # overwrite=True,
            )
            # pylint:enable=E1123
            gmpes.run(ds_name)
            # metrics = gmpes.run(ds_name)
            # print_results_box(metrics, 56, True)

    except Exception as e:  # pylint:disable=W0718
        print(
            f"Error for model {model_path}, dataset {ds_name}, pop {pop_size}, gen {num_gen}: {e}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--num_processes",
        type=int,
        default=4,
        help="Number of parallel processes",
    )
    parser.add_argument(
        "--pop_sizes",
        type=int,
        nargs="+",
        default=[10],
        help="List of population sizes to test",
    )
    parser.add_argument(
        "--num_gens",
        type=int,
        nargs="+",
        default=[20],
        help="List of number of generations to test",
    )

    args = parser.parse_args()

    models_768 = [
        r"\Embeddings\Size 768\Contriever\Base",
        r"\Embeddings\Size 768\Contriever\msmarco",
        r"\Embeddings\Size 768\DiffCSE\Bert base uncased",
        r"\Embeddings\Size 768\DiffCSE\RoBERTa base",
        r"\Embeddings\Size 768\E5",
        r"\Embeddings\Size 768\GTE",
        r"\Embeddings\Size 768\InfoCSE",
        r"\Embeddings\Size 768\SBERT-WK",
        r"\Embeddings\Size 768\WhiteningBERT",
        r"\Embeddings\Size 768\SimCSE",
        r"\Embeddings\Size 768\T5\3b single",
        r"\Embeddings\Size 768\T5\11b single",
        r"\Embeddings\Size 768\T5\Base",
        r"\Embeddings\Size 768\T5\Large",
    ]

    # Variants of the same models are not paired
    models_768_two_combo = [
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\E5",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\GTE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\InfoCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\E5",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\GTE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\InfoCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\E5",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\GTE",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\InfoCSE",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\E5",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\GTE",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\InfoCSE",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        ["\\Embeddings\\Size 768\\E5", "\\Embeddings\\Size 768\\GTE"],
        ["\\Embeddings\\Size 768\\E5", "\\Embeddings\\Size 768\\InfoCSE"],
        ["\\Embeddings\\Size 768\\E5", "\\Embeddings\\Size 768\\SBERT-WK"],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        ["\\Embeddings\\Size 768\\E5", "\\Embeddings\\Size 768\\SimCSE"],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        ["\\Embeddings\\Size 768\\E5", "\\Embeddings\\Size 768\\T5\\Base"],
        ["\\Embeddings\\Size 768\\E5", "\\Embeddings\\Size 768\\T5\\Large"],
        ["\\Embeddings\\Size 768\\GTE", "\\Embeddings\\Size 768\\InfoCSE"],
        ["\\Embeddings\\Size 768\\GTE", "\\Embeddings\\Size 768\\SBERT-WK"],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        ["\\Embeddings\\Size 768\\GTE", "\\Embeddings\\Size 768\\SimCSE"],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        ["\\Embeddings\\Size 768\\GTE", "\\Embeddings\\Size 768\\T5\\Base"],
        ["\\Embeddings\\Size 768\\GTE", "\\Embeddings\\Size 768\\T5\\Large"],
        [
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        ["\\Embeddings\\Size 768\\InfoCSE", "\\Embeddings\\Size 768\\SimCSE"],
        [
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        ["\\Embeddings\\Size 768\\SBERT-WK", "\\Embeddings\\Size 768\\SimCSE"],
        [
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        ["\\Embeddings\\Size 768\\SimCSE", "\\Embeddings\\Size 768\\T5\\Base"],
        [
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
    ]

    # Variants of the same models are not paired
    models_768_three_combo = [
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\E5",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\GTE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\InfoCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\E5",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\GTE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\InfoCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\GTE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\InfoCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\InfoCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\Base",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\E5",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\GTE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\InfoCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\E5",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\GTE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\InfoCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\GTE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\InfoCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\InfoCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\Contriever\\msmarco",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\GTE",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\InfoCSE",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\InfoCSE",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\Bert base uncased",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\GTE",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\InfoCSE",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\InfoCSE",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\DiffCSE\\RoBERTa base",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\InfoCSE",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\E5",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SBERT-WK",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\GTE",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\WhiteningBERT",
        ],
        [
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\InfoCSE",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\SBERT-WK",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
        [
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\11b single",
        ],
        [
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\Base",
        ],
        [
            "\\Embeddings\\Size 768\\WhiteningBERT",
            "\\Embeddings\\Size 768\\SimCSE",
            "\\Embeddings\\Size 768\\T5\\Large",
        ],
    ]

    models_1024 = [
        r"\Embeddings\Size 1024\BGE",
        r"\Embeddings\Size 1024\GTE",
        r"\Embeddings\Size 1024\InfoCSE",
        r"\Embeddings\Size 1024\SimCSE",
        r"\Embeddings\Size 1024\T5\3b single",
    ]

    models_1024_two_combo = [
        ["\\Embeddings\\Size 1024\\BGE", "\\Embeddings\\Size 1024\\GTE"],
        ["\\Embeddings\\Size 1024\\BGE", "\\Embeddings\\Size 1024\\InfoCSE"],
        ["\\Embeddings\\Size 1024\\BGE", "\\Embeddings\\Size 1024\\SimCSE"],
        [
            "\\Embeddings\\Size 1024\\BGE",
            "\\Embeddings\\Size 1024\\T5\\3b single",
        ],
        ["\\Embeddings\\Size 1024\\GTE", "\\Embeddings\\Size 1024\\InfoCSE"],
        ["\\Embeddings\\Size 1024\\GTE", "\\Embeddings\\Size 1024\\SimCSE"],
        [
            "\\Embeddings\\Size 1024\\GTE",
            "\\Embeddings\\Size 1024\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 1024\\InfoCSE",
            "\\Embeddings\\Size 1024\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 1024\\InfoCSE",
            "\\Embeddings\\Size 1024\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 1024\\SimCSE",
            "\\Embeddings\\Size 1024\\T5\\3b single",
        ],
    ]

    models_1024_three_combo = [
        [
            "\\Embeddings\\Size 1024\\BGE",
            "\\Embeddings\\Size 1024\\GTE",
            "\\Embeddings\\Size 1024\\InfoCSE",
        ],
        [
            "\\Embeddings\\Size 1024\\BGE",
            "\\Embeddings\\Size 1024\\GTE",
            "\\Embeddings\\Size 1024\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 1024\\BGE",
            "\\Embeddings\\Size 1024\\GTE",
            "\\Embeddings\\Size 1024\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 1024\\BGE",
            "\\Embeddings\\Size 1024\\InfoCSE",
            "\\Embeddings\\Size 1024\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 1024\\BGE",
            "\\Embeddings\\Size 1024\\InfoCSE",
            "\\Embeddings\\Size 1024\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 1024\\BGE",
            "\\Embeddings\\Size 1024\\SimCSE",
            "\\Embeddings\\Size 1024\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 1024\\GTE",
            "\\Embeddings\\Size 1024\\InfoCSE",
            "\\Embeddings\\Size 1024\\SimCSE",
        ],
        [
            "\\Embeddings\\Size 1024\\GTE",
            "\\Embeddings\\Size 1024\\InfoCSE",
            "\\Embeddings\\Size 1024\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 1024\\GTE",
            "\\Embeddings\\Size 1024\\SimCSE",
            "\\Embeddings\\Size 1024\\T5\\3b single",
        ],
        [
            "\\Embeddings\\Size 1024\\InfoCSE",
            "\\Embeddings\\Size 1024\\SimCSE",
            "\\Embeddings\\Size 1024\\T5\\3b single",
        ],
    ]

    all_paths = (
        models_768
        + models_1024
        + models_768_two_combo
        + models_1024_two_combo
        + models_768_three_combo
        + models_1024_three_combo
    )

    ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)

    datasets = ["sickr", "stsb", "sts12", "sts13", "sts14", "sts15", "sts16"]
    tasks = list(product(all_paths, datasets, args.pop_sizes, args.num_gens))

    process_tasks = [
        tasks[i :: args.num_processes] for i in range(args.num_processes)
    ]
    completed_tasks = Value("i", 0)

    def update_progress(_):
        with completed_tasks.get_lock():  # pylint: disable=W0640
            completed_tasks.value += 1  # pylint: disable=W0640

    with Pool(processes=args.num_processes) as pool:
        with tqdm(
            total=len(tasks),
            desc=f"Processing all tasks in parallel ({args.num_processes} processes)",
        ) as pbar:
            for i in range(args.num_processes):
                for task in process_tasks[i]:
                    pool.apply_async(
                        run_gmpes,
                        args=(task),
                        callback=update_progress,
                    )
            pool.close()

            while completed_tasks.value < len(tasks):
                pbar.n = completed_tasks.value
                pbar.refresh()
                time.sleep(0.1)
            pbar.n = len(tasks)
            pbar.refresh()
            pool.join()


ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
