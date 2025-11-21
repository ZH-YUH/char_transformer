# Char Transformer

This repository contains experiments with **character-level language models** on text8 datasets
It includes several Transformer variants and a LSTM model

The notebooks that are named "smallexperiment..." runs when we have limited computation power.
The notebooks that are named "transformer" and "lstm" runs when we have more computation power.

The .csv files contains the best parameters found by grid search from the small experiments

---

## Project Structure

```text
CHAR_TRANSFORMER/
├── data/
│   ├── text8_train.txt
│   └── text8_test.txt
│
├── models/
│   ├── __pycache__/
│   ├── models1.py            # best performing model copied from models.ipynb
│   ├── models2.py			  # second best performing model copied from models.ipynb
│   ├── models.ipynb          # every model we came up with
│   └── models_pytorch_variants.ipynb
│
├── util/
│   ├── __pycache__/
│   └── generation.py
│
├── transformer1.ipynb        # main Transformer training & evaluation notebook for model1
├── transformer2.ipynb        # main Transformer training & evaluation notebook for model2
├── lstm.ipynb                # LSTM training / evaluation
│
├── smallExperiment_lstm.ipynb
├── smallExperiment_varient_model.ipynb
│                             # model tuning and evaluation with limited resource
│
├── mini_grid_results_all_models.csv
├── lstm_round2_results.csv
├── lstm_smallExperiment_results.csv
├── second_round_results_all_models.csv
│                             # CSV logs of hyperparameter search and model comparisons
│
├── README.md                 # (this file)
├── requirements.txt
