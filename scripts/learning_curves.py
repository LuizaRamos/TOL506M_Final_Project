import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from mpl_toolkits.axes_grid1.axes_size import Fraction

from utils import plot_learning_curves

sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from tasks.task1 import train_from_scratch
from tasks.task2 import fine_tune_pretrained
from tasks.task3 import zero_shot_classification

def main ():
    parser = argparse.ArgumentParser(
        description="Plots the learning curve of each all tasks."
    )
    parser.add_argument(
        '--skip_training',
        action='store_true',
        help='skip training and only plot from existing results.'
    )

    args = parser.parse_args()

    config = Config()

    scratch_accs = []
    fine_tune_accs = []
    zero_shot_accs = []

    if not args.skip_training:
        # Train models with different data fraction
        for fraction in config.DATA_FRACTIONS:
            result_task1 = train_from_scratch(config=config, data_fraction=fraction)
            scratch_accs.append(result_task1['test_metrics']['accuracy'])
            result_task2 = fine_tune_pretrained(config=config, data_fraction=fraction)
            fine_tune_accs.append(result_task2['test_metrics']['accuracy'])

        result_task3 = zero_shot_classification(config=config)
        zero_shot_accs.append(result_task3['test_metrics']['accuracy'])

    else:
        # Loads from existing results -> FIX IT SO THAT IT GETS THE PROPER FILE
        for fraction in config.DATA_FRACTIONS:
            path_task1 = config.METRICS_DIR / f"task1_scratch_learning_curves_{fraction: .2f}.json" # alter to the proper name ...curves_{results_version_name}.json"
            with open(path_task1) as f:
                results_task1 = json.load(f)
                scratch_accs.append(results_task1['test_metrics']['accuracy'])

            path_task2 = config.METRICS_DIR / f"task2_fine_tuned_{fraction: .2f}.json" # alter to the proper name ...tuned_{results_version_name}.json
            with open(path_task2) as f:
                results_task2 = json.load(f)
                fine_tune_accs.append(results_task2['test_metrics']['accuracy'])

        path_task3 = config.RESULTS_DIR / f'task3_zero_shot_.json' # alter to the proper name ...shot_{results_version_name}.json
        with open(path_task3) as f:
            results_task3 = json.load(f)
            zero_shot_accs = results_task3['test_metrics']['accuracy']

    summary_data = {
        'Data Fraction': [f"{f:.0%}" for f in config.DATA_FRACTIONS],
        'Training from Scratch': [f"{acc:.2f}%" for acc in scratch_accs],
        'Fine-tuning Pretrained': [f"{acc:.2f}%" for acc in fine_tune_accs],
        'Zero-shot': [f"{zero_shot_accs:.2f}%"] * len(config.DATA_FRACTIONS)
    }

    df = pd.DataFrame(summary_data)

    # Save summary
    csv_path = config.METRICS_DIR / f"learning_curve_summary.json" # Add an increment in the name
    df.to_json(csv_path, index=False)
    print(f"Learning curve summary saved to {csv_path}.")

    plot_path = config.METRICS_DIR / f"learning_curve_plot.png"
    plot_learning_curves(
        data_fractions=config.DATA_FRACTIONS,
        scratch_accuracies=scratch_accs,
        finetune_accuracies=fine_tune_accs,
        zeroshot_accuracies=zero_shot_accs,
        save_path=str(plot_path)
    )
    print(f"Learning curve plot saved to {plot_path}.")

if __name__ == "__main__":
    main()