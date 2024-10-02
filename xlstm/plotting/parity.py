import datetime
from typing import Dict, List

import matplotlib.font_manager as font_manager  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa
import seaborn as sns

import wandb

# Set global plot style
plt.style.use(['science', 'no-latex', 'light'])
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'


def fetch_wandb_data(project_name: str, date_threshold_low: datetime.datetime, date_threshold_high: datetime.datetime) -> Dict[str, List[Dict]]:
    """Fetch and filter data from wandb."""
    api = wandb.Api()
    runs = api.runs(f"wandb_project/{project_name}")

    data = {'sigmoid': [], 'tanh': []}

    for run in runs:
        if datetime.datetime.strptime(run.created_at[:-1], "%Y-%m-%dT%H:%M:%S") > date_threshold_low and datetime.datetime.strptime(run.created_at[:-1], "%Y-%m-%dT%H:%M:%S") < date_threshold_high:
            config = run.config
            if (config.get('dataset', {}).get('kwargs', {}).get('synth_lang_type') == 'parity' and
                    config.get('model', {}).get('activation_func') in ['sigmoid', 'tanh']):
                history = run.scan_history()
                for row in history:
                    if "val_validation_SequenceAccuracy" in row and "_step" in row:
                        data[config['model']['activation_func']].append({
                            "seed": config.get('training', {}).get('seed'),
                            "step": row["_step"] * 200,
                            "accuracy": row["val_validation_SequenceAccuracy"]
                        })
    return data


def process_data(data: Dict[str, List[Dict]]) -> Dict[str, pd.DataFrame]:
    """Convert raw data to DataFrames."""
    return {
        'sigmoid': pd.DataFrame(data['sigmoid']).dropna(axis=0),
        'tanh': pd.DataFrame(data['tanh']).dropna(axis=0)
    }


def setup_plot():
    """Set up the plot figure and axes."""
    fig, ax = plt.subplots(figsize=(2.5, 1.5))
    return fig, ax


def plot_data(ax, dataframes: Dict[str, pd.DataFrame]):
    """Plot the data on the given axes."""
    colors = sns.color_palette("deep")
    for i, (activation, df) in enumerate([('[0, 1]', dataframes['sigmoid']), ('[-1, 1]', dataframes['tanh'])]):
        df.accuracy = (df.accuracy - 0.5) / 0.5
        grouped = df.groupby('step')
        mean = grouped['accuracy'].mean()
        std = grouped['accuracy'].std()

        ax.plot(mean.index, mean.values, label=activation, color=colors[i], linewidth=2)
        ax.fill_between(mean.index, mean.values - std.values, mean.values + std.values,
                        alpha=0.3, color=colors[i])


def configure_plot(ax):
    """Configure plot aesthetics."""
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Scaled Accuracy")
    ax.set_xlim(left=0, right=20000)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.arange(0.0, 1.1, 0.25))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title="Eigenvalue\nRange", loc='center left', bbox_to_anchor=(1, 0.5), handlelength=1, handleheight=0.7)


def save_plot(fig, filename_prefix: str):
    """Save the plot in multiple formats."""
    for fmt in ['pdf', 'png']:
        fig.savefig(f"{filename_prefix}.{fmt}", format=fmt, dpi=300, bbox_inches='tight')


def main():
    project_name = "xlstm-training"
    date_threshold_low = datetime.datetime(2024, 9, 1)
    date_threshold_high = datetime.datetime(2024, 9, 3)

    raw_data = fetch_wandb_data(project_name, date_threshold_low, date_threshold_high)
    processed_data = process_data(raw_data)

    fig, ax = setup_plot()
    plot_data(ax, processed_data)
    configure_plot(ax)

    plt.tight_layout()
    save_plot(fig, "parity_activation_function_comparison")
    plt.show()


if __name__ == "__main__":
    main()
    wandb.finish()
