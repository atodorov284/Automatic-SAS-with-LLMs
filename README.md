# 📚 Short Answer Grading Project

This project aims to automate short answer grading using NLP models, leveraging BERT-based scoring models and hyperparameter tuning techniques. The experiments incorporate LoRA configurations for fine-tuning sequence classification models. 

---

## 🗂️ Project Structure

- **`bert_scoring_model.py`**: Defines the BERT-based scoring model architecture, including the training and evaluation functions.
- **`experiment_runner.py`**: Handles the conducting of experiments, including model selection, hyperparameter tuning, and result logging.
- **`qwk.py`**: Defines the scoring metric used for this study. Code is taken from the Hewlett Foundation (Barbara et. al., 2012), as it uses the same dataset.
- **`short_answer_grading.ipynb`**: Jupyter Notebook for running interactive experiments and visualizations. This notebook allows a deep dive into the model performance and evaluation metrics.
- **`requirements.txt`**: Lists all necessary packages and dependencies required to set up and run the project.
---

## 🚀 Getting Started

### 1. Setting Up the Environment

1. **Clone the repository** (or download the files) and navigate to the project directory.
   
   ```bash
   git clone https://github.com/elisaklunder/Automatic-SAS-with-LLMs.git
   cd Automatic-SAS-with-LLMs
2. **Create a virtual environment (recommended) to prevent any potential dependency issues.**
    ```bash
    python3 -m venv .venv
3. **Activate the virtual environment:**
    - On Linux/macOS:
        ```bash
        source .venv/bin/activate
    - On Windows:
        ```bash
        .venv\Scripts\activate
4. **Install the dependencies listed in requirements.txt:**
    ```bash
    pip install -r requirements.txt
### 2. Running the Jupyter Notebook
The `short_answer_grading.ipynb` notebook is the primary file for running experiments interactively.
It contains the execution of all experiments of the study, along with the configurations, hyperparameters, results, and baselines.

1. **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
2. **Open `short_answer_grading.ipynb` and run the cells to execute the experiments.**
    This notebook provides step-by-step explanations for training and evaluating the model on short answer grading.
    > [!NOTE]
    > Some of the cells are commented, as they contain the training process and results, so it is easy to missclick them and rerun the whole training again. Uncomment them if you want to retrain or re-evaluate the models, by using the specified arguments (described further down).


## 📂 File Descriptions

- bert_scoring_model.py
    Contains the core implementation of the BERT-based model used for scoring short answers. It is a wrapper class for the BERT uncased model from `HuggingFace`. It includes methods for:
    - Loading pre-trained BERT models from transformers.
    - Fine-tuning for short answer scoring.
    - Model evaluation using metrics.
    - Loading already saved models, fine-tuned in this study.

- experiment_runner.py
    Manages the experiment workflow, integrating the BERT scoring model with various configurations and running tasks such as:
    - Full fine-tuning of the BERT uncased model.
    - LoRA (Low-Rank Adaptation) configurations for sequence classification.
    - Logging and saving experiment results.

- qwk.py
    This script contains the logic for computing the Quadratic Weighted Kappa score, taken by the code provided by the Hewlett Foundation.
    - To get the QWK score between two list of graders, you can call:

    `quadratic_weighted_kappa(rater_a: list[int], rater_b: list[int], min_rating: int = None, max_rating: int = None)`


- short_answer_grading.ipynb
    The main point of entry for this project, as it contains the conducting of experiments.. Key sections include:
    - Data exploration.
    - Augmenting prompts to include instructions (instruction-engineering).
    - Model training (fine-tuning) and evaluation routines.
    - Hyperparameter tuning setups.
    - Performance metrics based on the Quadratic Weighted Kappa score.
> [!NOTE]
> The notebook has extensive documentation in markdown and a step-by-step walkthrough for reproducibility.

- requirements.txt
    - Lists the dependencies required to run the project.

## 💡 Tips & Notes
- Make sure to activate the virtual environment each time you start a new session.
- Set `retrain=False` in order to avoid unnecessary retraining of the models. If you only wish to evaluate and get the scored from the saved models, set `evaluate=True`.
- The notebook includes sections for LoRA configuration in bert_scoring_model.py. You can adjust the parameters to experiment with different rank values, dropout, and target modules.

## Common Errors:

- `ImportError`: Make sure all dependencies are installed with pip install -r requirements.txt.
- Compatibility issues between Python and torch: Check the official PyTorch installation page for a compatible version based on your OS and hardware. We strongly recommend a virtual environment to run this project.

## 📄 License
This project is licensed under the MIT License.