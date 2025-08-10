# GMPES: Genetic Multi-Perspective Embedding Synthesis

GMPES is a genetic programming-based tool for synthesizing optimal combinations of sentence embeddings from multiple models to maximize Spearman correlation with ground truth similarity scores. It supports any number of embedding models, performs data splitting, optimization, evaluation, and result logging in JSON format. This tool is designed for natural language processing tasks like semantic similarity measurement, allowing users to fuse embeddings from models like SimCSE, T5, or others.

## Features
- Genetic programming optimization using DEAP to evolve mathematical expressions for embedding combination.
- Supports 1 or more embedding models with dynamic argument handling.
- Automatic training/testing data splitting (70/30 for single files, even split for multiple files, or pre-split if files contain "train"/"test").
- Protected mathematical operations for numerical stability (e.g., division by zero, negative logs).
- Baseline Spearman correlation computation for individual models.
- Atomic JSON result saving with order-independent model combination checking to avoid duplicates.
- Configurable parameters for population size, generations, crossover/mutation probabilities, size penalties, and more.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/AliApg/GMPES.git
   cd GMPES
   ```

2. Install dependencies (Python 3.8+ recommended):
   ```
   pip install numpy scipy scikit-learn sympy deap
   ```
   Note: DEAP is the core library for genetic programming. No additional installations are needed for basic use.

## Usage
The `GMPES` class can be imported and used in any Python script. It takes model embedding file paths and parameters, then runs optimization for a specified dataset.

### Basic Example
```python
from gmpes import GMPES, print_results_box

# Initialize with model files (list of paths per model)
gmpes = GMPES(
    model_files_path=[
        ["path/to/simcse embeddings"],  # Model 1 files
        ["path/to/t5 large embeddings"]       # Model 2 files
    ],
    save_folder_path="path/to/results_folder",
    population_size=100,
    num_generations=50,
    max_depth=10,
    overwrite=True
)

# Run optimization for a dataset (e.g., 'stsb')
metrics = gmpes.run("stsb")
print_results_box(metrics)
```

- **Input**: Model embedding files' folder path (filtered by dataset name internally) and configuration parameters.
- **Output**: A dictionary of metrics (best expression, train/test Spearman correlations for each model, elapsed time). Results are saved to `results.json` in the specified folder.
- **Customization**: Adjust parameters like `crossover_probability` or `size_penalty_threshold` for fine-tuning.

For looping over multiple datasets or models, implement your own loop:
```python
datasets = ["stsb", "sts12", "sts13"]
for ds in datasets:
    gmpes.run(ds)
```

## Testing
To test GMPES, use the provided `Embeddings` folder in the repository, which contains sample embedding files for models like SimCSE and T5. The folder structure is organized by embedding size (e.g., `Size 768`) and model type.

### Test Example
```python
gmpes = GMPES(
    model_files_path=[
        r"\Embeddings\Size 768\SimCSE embeddings",
        r"\Embeddings\Size 768\T5 embeddings\Large",
    ],
    save_folder_path= r"\Embeddings",
    population_size=50,
    num_generations=50,
)
print_results_box(metrics, 56, True)
```
Output:
```
╭────────────────────────────────────────────────────────────────────────────╮
│                              RESULTS SUMMARY                               │
├────────────────────────────────────────────────────────────────────────────┤
│Best Formula:       min(sin(sin(sub(min(m_1, m_2), m_2))), min(sin(min(si...│
├────────────────────────────────────────────────────────────────────────────┤
│                              TRAINING METRICS                              │
│SimCSE RoBERTa base                                                 0.853002│
│T5 large                                                            0.833824│
│best_fitness                                                      * 0.863041│
├────────────────────────────────────────────────────────────────────────────┤
│                              TESTING METRICS                               │
│SimCSE RoBERTa base                                                 0.856638│
│T5 large                                                            0.853606│
│test_correlation                                                  * 0.867393│
├────────────────────────────────────────────────────────────────────────────┤
│Time Elapsed:                                      22 minutes and 15 seconds│
╰────────────────────────────────────────────────────────────────────────────╯
```

- Run the example by replacing paths with your local repository location.
- Execute `gmpes.run("stsb")` (or other datasets) to perform optimization and save results.
- Verify `results.json` in the save folder for metrics.

## Input Format
The input to GMPES consists of folder paths containing pickle files for each model's embeddings. All embeddings must have the same dimensionality (e.g., 768) across models to ensure compatibility during combination.

### Pickle File Structure
Each pickle file (.pkl) should follow this naming convention and content format:
- **Filename**: `model-name(with no "_")_dsname.pkl`
  - Example: `SimCSE-RoBERTa base_stsb.pkl` (model name before underscore, dataset name after).
  - The model name part (before "_") is used to identify the model; avoid underscores in model names.
  - The dataset name part (after "_") is used to filter files for a specific dataset (e.g., "stsb").

- **Content** (Dictionary Keys):
  - `sentences_a_embeddings`: 2D NumPy array or list of lists (number of sentences × embedding size, e.g., 1000 × 768).
  - `sentences_b_embeddings`: Same as above for the second set of sentences.
  - `ground_truth_scores`: List[float] or 1D NumPy array of ground truth similarity scores (length equal to number of sentence pairs).

Example pickle file creation:
```python
import numpy as np
import pickle

data = {
    "sentences_a_embeddings": np.random.rand(1000, 768),
    "sentences_b_embeddings": np.random.rand(1000, 768),
    "ground_truth_scores": np.random.rand(1000)
}
with open("test model_test ds.pkl", "wb") as f:
    pickle.dump(data, f)
```

### Folder Structure
- Provide one or more folders containing pickle files for models.
- Example:
  ```
  embeddings_folder/
  ├── SimCSE_embeddings/
  │   ├── SimCSE-RoBERTa base_stsb.pkl
  │   ├── SimCSE-RoBERTa base_sts12.pkl
  │   └── ...
  ├── T5_embeddings/Large/
  │   ├── T5 large_stsb.pkl
  │   ├── T5 large_sts12.pkl
  │   └── ...
  ```
- Input to GMPES: Folder paths containing pickle files for each model's embeddings filtered by dataset name during runtime.
- Requirement: All models must produce embeddings of the same size (e.g., 768). If sizes differ, preprocessing is needed before input.

## Output
- **Console**: Progress messages, including dataset, models, parameters, and elapsed time.
  Example:
  ```
  ====================== Variables =======================
                      Dataset = stsb
              Model 1 = SimCSE RoBERTa base
                    Model 2 = T5 large
                    Model 3 = T5 base
                   Population size = 50
                Number of generations = 50
               Crossover probability = 0.7
                Mutation probability = 0.2
  ```
- **JSON File** (`results.json`): Structured results with model combinations (order-independent), experiments, and metrics.
  Example snippet:
  ```
  [
      [
          {
              "model_combination": {
                  "m_1": "SimCSE RoBERTa base",
                  "m_2": "T5 large"
              },
              "experiments": [
                  {
                      "parameters": {
                          "population_size": 50,
                          "num_generations": 50,
                          "crossover_probability": 0.7,
                          "mutation_probability": 0.2,
                          "size_penalty_coefficient": 0.0001
                      },
                      "results": {
                          "stsb_dataset": {
                              "best_expression": "min(sin(sin(sub(min(m_1, m_2), m_2))), min(sin(min(sin(m_2), m_1)), m_1))",
                              "train_metrics": {
                                  "SimCSE RoBERTa base": 0.853002,
                                  "T5 large": 0.833824,
                                  "best_fitness": 0.863041
                              },
                              "test_metrics": {
                                  "SimCSE RoBERTa base": 0.856638,
                                  "T5 large": 0.853606,
                                  "test_correlation": 0.867393
                              },
                              "time_elapsed": "22 minutes and 15 seconds"
                          },
                          ...
                      }
                  },
                  ...
              ]
          },
          ...
      ]
  ]
  ```

## Requirements
- Python 3.8+
- Libraries: `numpy`, `scipy`, `scikit-learn`, `sympy`, `deap`
- No internet access required; all operations are local.

## Contributing
Contributions are welcome! Please open an issue or pull request for bug fixes, features, or improvements. Ensure tests are added for new functionality.

## License
MIT License. See [LICENSE](LICENSE) for details.
