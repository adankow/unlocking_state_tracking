import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation as mad

import wandb


def fetch_wandb_data(project_name: str, date_threshold_low: datetime.datetime, date_threshold_high: datetime.datetime) -> Dict[str, List[Dict]]:
    """Fetch and filter data from wandb."""
    api = wandb.Api()
    runs = api.runs(f"wandb_project/{project_name}")

    data = {'positive': [], 'negative': []}

    for run in runs:
        if run.user.username == 'user':
            if datetime.datetime.strptime(run.created_at[:-1], "%Y-%m-%dT%H:%M:%S") > date_threshold_low and datetime.datetime.strptime(run.created_at[:-1], "%Y-%m-%dT%H:%M:%S") < date_threshold_high:
                config = run.config
                if run.state=='finished' and \
                config['dataset']['kwargs']['vocab_size'] == 12 and \
                config['dataset']['kwargs']['synth_lang_type'] == \
                'modular_arithmetic_with_brackets' and 'delta' in \
                config['model']['name'] and config['training']['lr'] == 0.001:
                    history = run.scan_history()
                    next(history)
                    last_step = history.rows[-1]
                    if last_step.get("val_validation_SequenceAccuracy") != None:
                        key = 'positive' if config['model']['sigmoid_scale'] == 1.0 else 'negative'
                        data[key].append({
                            "seed": config.get('training', {}).get('seed'),
                            "step": last_step["_step"] * 200,
                            "accuracy": last_step["val_validation_SequenceAccuracy"]
                        })
    return data


def process_data(data: Dict[str, List[Dict]]) -> Dict[str, pd.DataFrame]:
    """Convert raw data to DataFrames."""
    return {
        k: pd.DataFrame(v).dropna(axis=0) for k,v in data.items()
    }

def mean_confidence_interval(data, confidence=0.95):
    import scipy
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return np.max(a), m, np.median(a), h, np.std(a), mad(a)

def min_max_normalize(values, reference_value):
    """
    Normalize a list of values using min-max normalization around a specific reference value.

    :param values: pd.DataFrame containing the values
    :param reference_value: The value around which normalization is centered.
    :return: pd.DataFrame of normalized values.
    """
    min_value = 0 #min(values)
    max_value = 1 #max(values)

    # Normalize each value
    normalized_values = (values - reference_value) / (max_value - min_value)

    return normalized_values


def main():
    project_name = "xlstm-training"
    date_threshold_low = datetime.datetime(2024, 9, 26, 18)
    date_threshold_high = datetime.datetime(2024, 9, 28, 22)

    raw_data = fetch_wandb_data(project_name, date_threshold_low, date_threshold_high)
    processed_data = process_data(raw_data)
    print(processed_data)

    for n, data in processed_data.items():
        try:
            accs = (data['accuracy'] - 0.2) / 0.8
        except KeyError:
            continue
        #accs = min_max_normalize(data['accuracy'], 0.2)
        maxim, m, med, ci, std, mads = mean_confidence_interval(accs)
        print(maxim, n, m, med, std, mads)


if __name__ == "__main__":
    main()
    wandb.finish()
