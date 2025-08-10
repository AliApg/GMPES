"""
gmpes.py

This module provides classes for genetic programming-based embedding synthesis
and result logging. The `GMPES` class optimizes the combination of embeddings
from multiple models using genetic programming to maximize Spearman correlation
with ground truth scores. The `ResultSaver` class handles atomic JSON-based
logging of results, ensuring thread-safe and order-independent storage of model
combinations and experiment metrics.

Classes:
    ResultSaver: Manages atomic JSON-based logging of experiment results
        with conflict checking.
    GMPES: Performs genetic programming to optimize embedding combinations
        for multiple models.
"""

import json
import os
import pickle
import sys
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from deap import algorithms, base, creator, gp, tools
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sympy import (
    Float,
    Max,
    Min,
    Q,
    cos,
    exp,
    log,
    refine,
    sin,
    sqrt,
    symbols,
    sympify,
)

warnings.simplefilter("ignore")


def listdir_full(directory: Union[str, Path]) -> List[str]:
    """List all files in a directory with their full paths."""
    return [os.path.join(directory, item) for item in os.listdir(directory)]


def print_results_box(data, max_length=None, truncate_formula=False):
    """
    Prints a visually formatted results box with aligned metrics and formula display.

    The output features:
    - A bordered box with consistent right-alignment for all values
    - Smart handling of long formulas (wrapping or truncation with ellipsis)
    - Automatic width adjustment based on content or specified constraints
    - Clean display of training/testing metrics with fixed decimal precision

    Args:
        data (dict): Contains the following required keys:
            - best_expression (str): Mathematical formula/expression to display
            - time_elapsed (str): Duration string (e.g., "2 minutes 15 seconds")
            - train_metrics (dict): Metric names to values mapping for training
            - test_metrics (dict): Metric names to values mapping for testing

        max_length (int, optional): Strict maximum character limit for formula display.
                                   Forces box width adjustment when specified.
                                   Values are clamped between 30-250 characters.
                                   Default: None (auto-sized based on content)

        truncate_formula (bool): When True and formula exceeds display space:
                                - Truncates with "..." at max_length
                                When False:
                                - Wraps to multiple lines while preserving alignment
                                Default: False

    Returns:
        None: Output is printed directly to stdout

    Notes:
        - Minimum box width is 30 characters for readability
        - All numeric values are displayed with 6 decimal places
        - Empty data dictionary results in no output
    """

    if not data:
        return

    train_metrics = data["train_metrics"]
    test_metrics = data["test_metrics"]

    train_best = max(train_metrics.values())
    test_best = max(test_metrics.values())

    marked_train_metrics = {
        k: f"* {v:.6f}" if v == train_best else f"{v:.6f}"
        for k, v in train_metrics.items()
    }

    marked_test_metrics = {
        k: f"* {v:.6f}" if v == test_best else f"{v:.6f}"
        for k, v in test_metrics.items()
    }

    max_label_len = max(
        len("Best Formula:"),
        len("Time Elapsed:"),
        max(len(k) for k in train_metrics),
        max(len(k) for k in test_metrics),
    )

    max_value_len = max(
        len(data["time_elapsed"]),
        max(
            len(f"* {v:.6f}" if v == train_best else f"{v:.6f}")
            for v in train_metrics.values()
        ),
        max(
            len(f"* {v:.6f}" if v == test_best else f"{v:.6f}")
            for v in test_metrics.values()
        ),
    )

    required_content_width = max_value_len + 3
    min_width = 30
    max_length = min(
        max(required_content_width, max_length if max_length else 0), 250
    )

    if max_length is not None:
        formula_required_width = max_label_len + max_length + 3
        box_width = max(
            min_width, required_content_width, formula_required_width
        )
    else:
        box_width = max(min_width, required_content_width)

    formula_available_width = box_width - max_label_len - 3

    top_border = "╭" + "─" * (box_width - 2) + "╮"
    divider = "├" + "─" * (box_width - 2) + "┤"
    bottom_border = "╰" + "─" * (box_width - 2) + "╯"

    print(top_border)
    print(f"│{'RESULTS SUMMARY'.center(box_width - 2)}│")
    print(divider)

    label = "Best Formula:"
    formula = data["best_expression"]

    if max_length is not None:
        formula_available_width = min(formula_available_width, max_length)

    if truncate_formula and len(formula) > formula_available_width:
        truncated = formula[: formula_available_width - 3] + "..."
        print(
            f"│{label.ljust(max_label_len)} {truncated.rjust(formula_available_width)}│"
        )
    else:
        formula_lines = [
            formula[i : i + formula_available_width]
            for i in range(0, len(formula), formula_available_width)
        ]
        print(
            f"│{label.ljust(max_label_len)} {formula_lines[0].rjust(formula_available_width)}│"
        )
        for line in formula_lines[1:]:
            print(
                f"│{' ' * max_label_len} {line.rjust(formula_available_width)}│"
            )

    print(divider)

    print(f"│{'TRAINING METRICS'.center(box_width - 2)}│")
    for model, score in train_metrics.items():
        value = marked_train_metrics[model]
        print(
            f"│{model.ljust(max_label_len)} {value.rjust(box_width - max_label_len - 3)}│"
        )

    print(divider)

    print(f"│{'TESTING METRICS'.center(box_width - 2)}│")
    for model, score in test_metrics.items():
        value = marked_test_metrics[model]
        print(
            f"│{model.ljust(max_label_len)} {value.rjust(box_width - max_label_len - 3)}│"
        )

    print(divider)

    label = "Time Elapsed:"
    time_str = data["time_elapsed"]
    time_lines = [
        time_str[i : i + box_width - max_label_len - 3]
        for i in range(0, len(time_str), box_width - max_label_len - 3)
    ]
    print(
        f"│{label.ljust(max_label_len)} {time_lines[0].rjust(box_width - max_label_len - 3)}│"
    )
    for line in time_lines[1:]:
        print(
            f"│{' ' * max_label_len} {line.rjust(box_width - max_label_len - 3)}│"
        )

    print(bottom_border)


class ResultSaver:
    """
    A class for atomically saving and managing experiment results in a JSON file.

    This class handles the storage of genetic programming experiment results,
    including model combinations, parameters, and performance metrics, in a
    JSON file. It ensures atomic writes to prevent file corruption, supports
    order-independent model combination checking, and skips existing results
    unless overwritten.

    Attributes:
        filepath (str): Path to the JSON file for storing results.

    Methods:
        _initialize_file: Creates an empty JSON array if the file doesn't exist.
        _find_or_create_combo: Finds or creates a model combination entry in the JSON data.
        _find_or_create_experiment: Finds or creates an experiment entry for given parameters.
        _safe_writer: Opens the file with retries to handle Windows file locking.
        _atomic_write: Performs atomic writes using temporary files to ensure data integrity.
        check_exists: Checks if results for a model combination and dataset exist.
        log_dataset: Saves dataset results with conflict checking and optional overwrite.
    """

    def __init__(self, filepath: Union[str, Path]):
        """
        Initialize ResultSaver for managing JSON-based result logging.

        Args:
            filepath (Union[str, Path]): Path to the JSON file for saving results.
        """

        self.filepath = str(Path(filepath))
        self._initialize_file()

    def _initialize_file(self):
        """Create empty JSON array if file doesn't exist."""

        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _find_or_create_combo(
        self, data: List[Dict], models: Dict[str, str]
    ) -> Dict:
        """Find model combo or create new entry (order-independent)."""

        model_set = set(models.values())
        for entry in data:
            if set(entry["model_combination"].values()) == model_set:
                return entry
        new_entry = {"model_combination": models, "experiments": []}
        data.append(new_entry)
        return new_entry

    def _find_or_create_experiment(
        self, combo_entry: Dict, parameters: Dict
    ) -> Dict:
        """Find parameter set or create new experiment."""

        for exp in combo_entry["experiments"]:  # pylint: disable=W0621
            if exp["parameters"] == parameters:
                return exp
        new_exp = {
            "parameters": parameters,
            "results": {},
            "timestamp": datetime.now().isoformat(),
        }
        combo_entry["experiments"].append(new_exp)
        return new_exp

    def _safe_writer(self, mode: str = "w"):
        """Handle Windows file locking with retries."""

        max_retries = 3
        retry_delay = 0.1
        for attempt in range(max_retries):
            try:
                return open(self.filepath, mode, encoding="utf-8")
            except PermissionError:
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay)

    def _atomic_write(self, data: List[Dict]) -> Literal[True] | None:
        """Atomic write with Windows-safe temp file handling."""

        temp_dir = os.path.dirname(self.filepath) or "."
        for attempt in range(3):
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    dir=temp_dir,
                    delete=False,
                    encoding="utf-8",
                    prefix=".tmp_",
                    suffix=".json",
                ) as tmp:
                    json.dump(data, tmp, indent=4)
                    tmp.flush()
                    os.fsync(tmp.fileno())
                    tmp_name = tmp.name
                try:
                    os.replace(tmp_name, self.filepath)
                    return True
                except PermissionError:
                    time.sleep(0.1)
                    os.replace(tmp_name, self.filepath)
                    return True
            except PermissionError as e:
                if os.path.exists(tmp_name):
                    try:
                        os.unlink(tmp_name)
                    except:  # pylint: disable=W0702
                        pass
                if attempt == 2:
                    raise RuntimeError(
                        "Failed to write after 3 attempts."
                    ) from e
            except Exception as e:
                if os.path.exists(tmp_name):
                    try:
                        os.unlink(tmp_name)
                    except:  # pylint: disable=W0702
                        pass
                raise e

    def check_exists(
        self, models: Dict[str, str], parameters: Dict, ds_key: str
    ) -> bool:
        """Check if dataset results already exist for the model combination (order-independent)."""

        if not os.path.exists(self.filepath):
            return False
        with self._safe_writer("r") as f:  # type: ignore
            data = json.load(f)

        input_model_set = set(models.values())

        for entry in data:
            entry_model_set = set(entry["model_combination"].values())
            if entry_model_set == input_model_set:
                for exp in entry["experiments"]:  # pylint: disable=W0621
                    if exp["parameters"] == parameters:
                        return ds_key in exp["results"]
        return False

    def log_dataset(
        self,
        models: Dict[str, str],
        parameters: Dict,
        ds_name: str,
        metrics: Dict,  # pylint: disable=W0621
        overwrite: bool = False,
    ) -> bool:
        """
        Atomically save single dataset results with conflict checking.

        Args:
            models: Dictionary with model names (e.g., {"m_1": name1, "m_2": name2, ...}).
            parameters: Dictionary of GA parameters.
            ds_name: Dataset name (e.g., "stsb").
            metrics: Pre-calculated metrics dictionary.
            overwrite: Whether to overwrite existing results.

        Returns:
            bool: True if saved, False if skipped (existing complete).
        """

        for attempt in range(3):
            try:
                with self._safe_writer("r") as f:  # type: ignore
                    data = json.load(f)
                break
            except json.JSONDecodeError:
                if attempt == 2:
                    raise
                time.sleep(0.1)

        combo_entry = self._find_or_create_combo(data, models)
        experiment = self._find_or_create_experiment(combo_entry, parameters)
        ds_key = f"{ds_name}_dataset"
        if ds_key in experiment["results"] and not overwrite:
            return False
        experiment["results"][ds_key] = metrics
        experiment["timestamp"] = datetime.now().isoformat()
        return self._atomic_write(data)  # type: ignore


class GMPES:
    """
    Genetic Multi-Perspective Embedding Synthesis (GMPES) for optimizing embedding combinations.

    This class uses genetic programming (via DEAP) to optimize the combination of sentence
    embeddings from multiple models to maximize Spearman correlation with ground truth scores.
    It supports any number of models, splits data into training and testing sets, evaluates
    expressions, and saves results to a JSON file using ResultSaver. No visualizations are
    generated, and results are stored with order-independent model
    combination checking.

    Attributes:
        model_files (List[List[Union[str, Path]]]): List of file paths for each model's embeddings.
        save_folder_path (str): Path to save results JSON file.
        population_size (int): Size of the genetic programming population.
        num_generations (int): Number of generations for genetic programming.
        crossover_probability (float): Probability of crossover in genetic programming.
        mutation_probability (float): Probability of mutation in genetic programming.
        size_penalty_coefficient (float): Penalty coefficient for expression size.
        decimal (int): Decimal places for metrics output.
        line_length (int): Line length for console output.
        max_depth (int): Maximum depth of genetic programming trees.
        overwrite (bool): Whether to overwrite existing results.
        models (Dict[str, str]): Dictionary mapping model indices (e.g., 'm_1') to names.
        result_saver (ResultSaver): Instance for saving results to JSON.
        data_splits (Dict[str, Dict[str, Any]]): Training and testing data splits for each model.
        pset (gp.PrimitiveSet): DEAP primitive set for genetic programming.
        toolbox (base.Toolbox): DEAP toolbox for genetic programming operations.

    Methods:
        _setup_gp: Sets up DEAP genetic programming components for n models.
        _protected_div: Protected division operation for numerical stability.
        _protected_log: Protected logarithm operation for numerical stability.
        _protected_exp: Protected exponential operation for numerical stability.
        _protected_sqrt: Protected square root operation for numerical stability.
        _protected_inv: Protected multiplicative inverse operation.
        _protected_max: Protected element-wise maximum operation.
        _protected_min: Protected element-wise minimum operation.
        _protected_mean: Protected mean operation for aggregation.
        _filter_by_name: Filters file paths by a pattern in their base names.
        _load_pickle_files: Loads and concatenates data from pickle files.
        _train_test_splitting: Splits data for multiple models into training and testing sets.
        _split_data: Filters and splits data for a specific dataset.
        _evaluate: Evaluates an individual's fitness using training data.
        _test_performance: Evaluates the best expression on test data.
        _calculate_spearman: Calculates Spearman correlation for baseline embeddings.
        _report: Generates metrics dictionary for JSON saving.
        run: Runs genetic programming optimization for a dataset and saves results.
    """

    def __init__(
        self,
        model_files_path: List[Union[str, Path]],
        save_folder_path: Union[str, Path] = sys.path[0],
        population_size: int = 50,
        num_generations: int = 50,
        crossover_probability: float = 0.7,  # TODO: Change to: 0.8
        mutation_probability: float = 0.2,  # TODO: Change to: 0.15
        tournsize: Optional[int] = None,
        size_penalty_threshold: int = 15,  # TODO: Change to: 25
        size_penalty_coefficient: float = 0.0001,  # TODO: Change to: 0.0005
        max_depth: int = 7,  # TODO: Change to: 10
        decimal: int = 6,
        line_length: int = 56,
        overwrite: bool = False,
    ):
        """
        Initialize Genetic Multi-Perspective Embedding Synthesis (GMPES) for
        optimizing embedding combinations.

        Args:
            model_files_path (List[Union[str, Path]]): List of file path(s)
                for each model's embeddings.
            save_folder_path (Union[str, Path]): Path to save results JSON file.
            population_size (int, optional): Size of the GP population. Defaults to 50.
            num_generations (int, optional): Number of GP generations. Defaults to 50.
            crossover_probability (float, optional): Probability of crossover. Defaults to 0.7. # TODO: Change to: 0.8
            mutation_probability (float, optional): Probability of mutation. Defaults to 0.2. # TODO: Change to: 0.15
            tournsize (int | None, optional): Number of individuals for tournament selection.
                If None, set to max(3, int(log2(population_size))). Defaults to None.
            size_penalty_threshold (int, optional): Threshold for applying size penalty to
                expressions longer than this value. Defaults to 15. # TODO: Change to: 25
            size_penalty_coefficient (float, optional): Penalty coefficient for expression size.
                Defaults to 0.0001. # TODO: Change to: 0.0005
            max_depth (int, optional): Maximum depth of GP trees. Defaults to 7. # TODO: Change to: 10
            decimal (int, optional): Decimal places for metrics output. Defaults to 6.
            line_length (int, optional): Line length for console output. Defaults to 56.
            overwrite (bool, optional): Whether to overwrite existing results. Defaults to False.
        """

        self.model_files_path = [listdir_full(f) for f in model_files_path]
        self.save_folder_path = str(Path(save_folder_path))
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.tournsize = (
            tournsize
            if tournsize is not None
            else max(3, int(np.log2(self.population_size)))
        )
        self.size_penalty_threshold = size_penalty_threshold
        self.size_penalty_coefficient = size_penalty_coefficient
        self.max_depth = max_depth
        self.decimal = decimal
        self.line_length = line_length
        self.overwrite = overwrite

        self.models = {
            f"m_{i+1}": str(Path(files[0]).stem)
            .split("_", maxsplit=1)[0]
            .replace("-", " ")
            for i, files in enumerate(self.model_files_path)
        }

        self.result_saver = ResultSaver(
            os.path.join(self.save_folder_path, "results.json")
        )
        self.data_splits = None
        self.pset = None
        self.toolbox = None
        self._setup_gp()

    def _setup_gp(self):
        """Set up DEAP genetic programming components for n models."""

        self.pset = gp.PrimitiveSet("MAIN", len(self.model_files_path))
        for i in range(len(self.model_files_path)):
            self.pset.renameArguments(**{f"ARG{i}": f"m_{i+1}"})

        # Core arithmetic
        self.pset.addPrimitive(np.add, 2, name="add")
        self.pset.addPrimitive(np.subtract, 2, name="sub")
        self.pset.addPrimitive(np.multiply, 2, name="mul")
        self.pset.addPrimitive(self._protected_div, 2, name="div")
        self.pset.addPrimitive(self._protected_inv, 1, name="inv")

        # Nonlinear transforms
        self.pset.addPrimitive(self._protected_log, 1, name="log")
        self.pset.addPrimitive(self._protected_exp, 1, name="exp")
        self.pset.addPrimitive(self._protected_sqrt, 1, name="sqrt")
        self.pset.addPrimitive(np.sin, 1, name="sin")
        self.pset.addPrimitive(np.cos, 1, name="cos")

        # Conditional/aggregation
        self.pset.addPrimitive(self._protected_max, 2, name="max")
        self.pset.addPrimitive(self._protected_min, 2, name="min")
        self.pset.addPrimitive(self._protected_mean, 1, name="mean")

        # Random constant generation
        self.pset.addEphemeralConstant(
            "rand_const", lambda: np.random.uniform(-1, 1)
        )

        creator.create("FitnessMax", base.Fitness, weights=(1.0, 0.5))
        creator.create(
            "Individual",
            gp.PrimitiveTree,
            fitness=creator.FitnessMax,  # type:ignore pylint:disable=E1101
        )

        self.toolbox = base.Toolbox()
        self.toolbox.register(
            "expr",
            gp.genHalfAndHalf,
            pset=self.pset,
            min_=1,
            max_=self.max_depth,
        )
        self.toolbox.register(
            "individual",
            tools.initIterate,
            creator.Individual,  # type:ignore pylint:disable=E1101
            self.toolbox.expr,  # type:ignore pylint:disable=E1101
        )
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual,  # type:ignore pylint:disable=E1101
        )
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register(
            "select", tools.selTournament, tournsize=self.tournsize
        )
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register(
            "mutate",
            gp.mutUniform,
            expr=self.toolbox.expr_mut,  # type:ignore pylint:disable=E1101
            pset=self.pset,
        )
        self.toolbox.decorate(
            "mate",
            gp.staticLimit(
                key=lambda ind: ind.height, max_value=self.max_depth
            ),
        )
        self.toolbox.decorate(
            "mutate",
            gp.staticLimit(
                key=lambda ind: ind.height, max_value=self.max_depth
            ),
        )

    def _protected_div(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Protected division with protection against divide-by-zero."""

        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(y) < 1e-6, 1.0, np.divide(x, y))

    def _protected_log(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Protected logarithm with near-zero input handling."""

        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(np.abs(x) < 1e-6, 0.0, np.log(np.abs(x) + 1e-6))

    def _protected_exp(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Protected exponential with input clipping to avoid overflow."""

        return np.exp(np.clip(x, -20, 20))

    def _protected_sqrt(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Protected square root with negative input handling."""

        return np.sqrt(np.abs(x) + 1e-6)

    def _protected_inv(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Protected multiplicative inverse (1/x)."""

        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(x) < 1e-6, 0.0, 1.0 / x)

    def _protected_max(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Element-wise maximum with NaN protection."""

        return np.maximum(np.nan_to_num(x), np.nan_to_num(y))

    def _protected_min(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Element-wise minimum with NaN protection."""

        return np.minimum(np.nan_to_num(x), np.nan_to_num(y))

    def _protected_mean(
        self, args: List[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        """Protected mean across axis 0 (handles empty inputs)."""

        stacked = np.stack(args)
        return (
            np.mean(stacked, axis=0)
            if len(stacked) > 0
            else np.zeros_like(args[0])
        )

    def _filter_by_name(
        self, pattern: str, file_list: List[Union[str, Path]]
    ) -> List[Union[str, Path]]:
        """Filter a list of file paths by a pattern in their base names."""

        return [
            file for file in file_list if pattern in os.path.basename(file)
        ]

    def _load_pickle_files(
        self, file_paths: List[Union[str, Path]]
    ) -> Dict[str, np.ndarray]:
        """Load and concatenate data from multiple pickle files."""

        sentences_a_embeddings = []
        sentences_b_embeddings = []
        ground_truth_scores = []

        for file_path in file_paths:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            sentences_a_embeddings.append(data["sentences_a_embeddings"])
            sentences_b_embeddings.append(data["sentences_b_embeddings"])
            ground_truth_scores.append(data["ground_truth_scores"])

        return {
            "sentences_a_embeddings": np.concatenate(sentences_a_embeddings),
            "sentences_b_embeddings": np.concatenate(sentences_b_embeddings),
            "ground_truth_scores": np.concatenate(ground_truth_scores),
        }

    def _train_test_splitting(
        self, *models: Union[str, Path]
    ) -> Dict[str, Dict[str, Any]]:
        """Split data for multiple models into training and testing sets."""

        result = {}
        for model, model_key in zip(models, self.models.keys()):
            model_name = self.models[model_key]  # Use name from self.models
            train_data = {}
            test_data = {}
            if self._filter_by_name("train", model):  # type: ignore
                train_data = self._load_pickle_files(
                    self._filter_by_name("train", model)  # type: ignore
                )
                test_data = self._load_pickle_files(
                    self._filter_by_name("test", model)  # type: ignore
                )
            else:
                model_files = (
                    [model] if isinstance(model, (str, Path)) else model
                )
                len_files = len(model_files)
                if len_files > 1:
                    train_files = model_files[: len_files // 2]
                    test_files = model_files[len_files // 2 :]
                    train_data = self._load_pickle_files(train_files)
                    test_data = self._load_pickle_files(test_files)
                else:
                    data = self._load_pickle_files(model_files)
                    split_point = int(len(data["ground_truth_scores"]) * 0.7)
                    train_data = {
                        "sentences_a_embeddings": data[
                            "sentences_a_embeddings"
                        ][:split_point],
                        "sentences_b_embeddings": data[
                            "sentences_b_embeddings"
                        ][:split_point],
                        "ground_truth_scores": data["ground_truth_scores"][
                            :split_point
                        ],
                    }
                    test_data = {
                        "sentences_a_embeddings": data[
                            "sentences_a_embeddings"
                        ][split_point:],
                        "sentences_b_embeddings": data[
                            "sentences_b_embeddings"
                        ][split_point:],
                        "ground_truth_scores": data["ground_truth_scores"][
                            split_point:
                        ],
                    }
            result[model_name] = {"train": train_data, "test": test_data}
        return result

    def _split_data(self, ds_name: str):
        """Split data for all models into training and testing sets for a given dataset."""

        filtered_files = [
            self._filter_by_name(ds_name, files)  # type: ignore
            for files in self.model_files_path
        ]
        self.data_splits = self._train_test_splitting(*filtered_files)  # type: ignore

    def _evaluate(self, individual) -> tuple[float, float]:
        """Evaluate an individual's fitness using training data."""

        try:
            func = self.toolbox.compile(  # type:ignore pylint:disable=E1101
                expr=individual
            )
            cos_similarities = []
            for sample in zip(
                *[
                    self.data_splits[model_name]["train"][key]  # type: ignore
                    for model_name in self.data_splits  # type: ignore
                    for key in [
                        "sentences_a_embeddings",
                        "sentences_b_embeddings",
                    ]
                ]
            ):
                a_embeds = sample[::2]
                b_embeds = sample[1::2]
                a_combined = func(*a_embeds)
                b_combined = func(*b_embeds)
                norm_a = np.linalg.norm(a_combined)
                norm_b = np.linalg.norm(b_combined)
                cos_sim = (
                    0.0
                    if norm_a == 0 or norm_b == 0
                    else np.dot(a_combined, b_combined) / (norm_a * norm_b)
                )
                cos_similarities.append(cos_sim)

            ground_truth = next(iter(self.data_splits.values()))["train"][  # type: ignore
                "ground_truth_scores"
            ]
            spearman_corr, _ = spearmanr(
                cos_similarities, ground_truth[: len(cos_similarities)]
            )
            if np.isnan(spearman_corr):  # type: ignore
                return (0.0, 0.0)

            size_penalty = (
                len(individual) * self.size_penalty_coefficient
                if len(individual) > self.size_penalty_threshold
                else 0
            )
            return (max(0, spearman_corr - size_penalty), spearman_corr)  # type: ignore
        except Exception:  # pylint: disable=W0718
            return (0.0, 0.0)

    def _test_performance(self, best_expr) -> float:
        """Evaluate the best expression on test data."""

        best_func = self.toolbox.compile(  # type:ignore pylint:disable=E1101
            expr=best_expr
        )
        cos_similarities = []
        nan_count = 0

        for sample in zip(
            *[
                self.data_splits[model_name]["test"][key]  # type: ignore
                for model_name in self.data_splits  # type: ignore
                for key in ["sentences_a_embeddings", "sentences_b_embeddings"]
            ]
        ):
            try:
                with np.errstate(all="ignore"):
                    a_embeds = sample[::2]
                    b_embeds = sample[1::2]
                    a_combined = best_func(*a_embeds)
                    b_combined = best_func(*b_embeds)
                    a_combined = np.nan_to_num(
                        a_combined, nan=0.0, posinf=1e6, neginf=-1e6
                    )
                    b_combined = np.nan_to_num(
                        b_combined, nan=0.0, posinf=1e6, neginf=-1e6
                    )
                    if (
                        np.linalg.norm(a_combined) == 0
                        or np.linalg.norm(b_combined) == 0
                    ):
                        cos_sim = 0.0
                    else:
                        cos_sim = cosine_similarity(
                            a_combined.reshape(1, -1),
                            b_combined.reshape(1, -1),
                        )[0][0]
                    cos_similarities.append(cos_sim)
            except Exception:  # pylint: disable=W0718
                nan_count += 1
                cos_similarities.append(0.0)

        if nan_count:
            print(
                f"Warning: {nan_count}/"
                f"{len(next(iter(self.data_splits.values()))['test']['ground_truth_scores'])}"  # type: ignore
                "samples produced invalid values"
            )
        ground_truth = next(iter(self.data_splits.values()))["test"][  # type: ignore
            "ground_truth_scores"
        ]
        spearman_corr, _ = spearmanr(
            cos_similarities, ground_truth[: len(cos_similarities)]
        )
        return spearman_corr  # type: ignore

    def _calculate_spearman(
        self, embeddings_data: Dict[str, np.ndarray]
    ) -> float:
        """Calculate Spearman correlation for baseline embeddings."""

        cos_similarities = np.array(
            [
                cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]
                for a, b in zip(
                    embeddings_data["sentences_a_embeddings"],
                    embeddings_data["sentences_b_embeddings"],
                )
            ]
        )
        return spearmanr(
            cos_similarities, embeddings_data["ground_truth_scores"]
        )[
            0
        ]  # type: ignore

    def _report(self, best_expr, test_corr: float) -> Dict[str, Any]:
        """Generate metrics dictionary for JSON saving."""

        converter = {
            "sub": lambda x, y: x - y,
            "add": lambda x, y: x + y,
            "mul": lambda x, y: x * y,
            "div": lambda x, y: x / y,
            "inv": lambda x: 1 / x,
            "log": log,
            "exp": exp,
            "sqrt": sqrt,
            "sin": sin,
            "cos": cos,
            "max": Max,
            "min": Min,
            "mean": lambda *arguments: sum(arguments) / len(arguments),
            "rand_const": lambda: Float(np.random.uniform(-1, 1)),
        }

        def expression_to_sympy(expr):
            try:
                syms = symbols(
                    " ".join(
                        [f"m_{i+1}" for i in range(len(self.model_files_path))]
                    ),
                    real=True,
                )
                expr_str = str(expr)
                return str(
                    refine(
                        sympify(expr_str, locals=converter),
                        Q.positive(syms)  # pylint: disable=E1121
                        & Q.real(syms),  # pylint: disable=E1121
                    )
                )
            except Exception as e:  # pylint: disable=W0718
                return f"Failed to simplify => {str(e)}"

        train_metrics = {
            model_name: round(
                self._calculate_spearman(
                    self.data_splits[model_name]["train"]  # type: ignore
                ),
                self.decimal,
            )
            for model_name in self.data_splits  # type: ignore
        }
        train_metrics["best_fitness"] = round(
            best_expr.fitness.values[1], self.decimal
        )

        test_metrics = {
            model_name: round(
                self._calculate_spearman(self.data_splits[model_name]["test"]),  # type: ignore
                self.decimal,
            )
            for model_name in self.data_splits  # type: ignore
        }
        test_metrics["test_correlation"] = round(test_corr, self.decimal)

        return {
            "best_expression": expression_to_sympy(best_expr),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }

    def run(self, ds_name: str) -> Dict[str, Any]:
        """Run GP optimization for a dataset and save results."""

        parameters = {
            "population_size": self.population_size,
            "num_generations": self.num_generations,
            "crossover_probability": self.crossover_probability,
            "mutation_probability": self.mutation_probability,
            "size_penalty_coefficient": self.size_penalty_coefficient,
        }
        ds_key = f"{ds_name}_dataset"
        if not self.overwrite and self.result_saver.check_exists(
            self.models, parameters, ds_key
        ):
            print(
                f"Skipped existing result for dataset '{ds_name}' with models "
                f"{', '.join(self.models.values())} with the same experiment parameters."
                " To overwrite, set 'overwrite' to True."
            )
            return {}

        self._split_data(ds_name)
        start_time = time.time()

        print(
            "",
            " Variables ".center(self.line_length, "="),
            f"Dataset = {ds_name}".center(self.line_length),
            *[
                f"Model {i+1} = {name}".center(self.line_length)
                for i, name in enumerate(self.models.values())
            ],
            f"Population size = {self.population_size}".center(
                self.line_length
            ),
            f"Number of generations = {self.num_generations}".center(
                self.line_length
            ),
            f"Crossover probability = {self.crossover_probability}".center(
                self.line_length
            ),
            f"Mutation probability = {self.mutation_probability}".center(
                self.line_length
            ),
            sep="\n",
            end="\n\n",
        )

        pop = self.toolbox.population(  # type:ignore pylint:disable=E1101
            n=self.population_size
        )
        hall_of_fame = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        algorithms.eaSimple(
            pop,
            self.toolbox,
            cxpb=self.crossover_probability,
            mutpb=self.mutation_probability,
            ngen=self.num_generations,
            stats=stats,
            halloffame=hall_of_fame,
            verbose=True,
        )

        best_expr = hall_of_fame[0]
        test_corr = self._test_performance(best_expr)
        metrics = self._report(best_expr, test_corr)  # pylint: disable=W0621

        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        hours, minutes, seconds = int(hours), int(minutes), int(seconds)
        msg_parts = []
        if hours > 0:
            msg_parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
        if minutes > 0:
            msg_parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
        if seconds > 0 or not msg_parts:
            msg_parts.append(f"{seconds} second{'s' if seconds > 1 else ''}")
        msg = (
            ", ".join(msg_parts[:-1]) + ", and " + msg_parts[-1]
            if len(msg_parts) > 2
            else " and ".join(msg_parts)
        )
        metrics["time_elapsed"] = msg

        self.result_saver.log_dataset(
            self.models, parameters, ds_name, metrics, self.overwrite
        )
        print(f"\nGenetic programming completed in {msg}.")
        return metrics


if __name__ == "__main__":

    # Algorithm Testing

    gmpes = GMPES(
        model_files_path=[
            sys.path[0] + r"\Embeddings\Size 768\SimCSE embeddings",
            sys.path[0] + r"\Embeddings\Size 768\T5 embeddings\Large",
        ],
        population_size=5,
        num_generations=5,
    )
    metrics = gmpes.run("stsb")
    print(metrics)
