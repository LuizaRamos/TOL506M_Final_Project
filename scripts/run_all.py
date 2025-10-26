import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from tasks.task1 import train_from_scratch
from tasks.task2 import fine_tune_pretrained
from tasks.task3 import zero_shot_classification

def main():
    parser = argparse.ArgumentParser(
        description='Run all tasks for Wildlife classification.'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='animals10',
        choices=['animals10', 'oxford_pet', 'animals10n'],
        help='Dataset name'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Path to the data directory'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run all tasks for quick experiment with reduced epochs'
    )

    args = parser.parse_args()

    # Create config
    config = Config()
    config.DATASET = args.dataset

    if args.data_path:
        config.DATASET_PATH = Path(args.data_path)

    # Reduce epochs for quick testing
    if args.quick:
        config.SCRATCH_EPOCHS = 5
        config.FINETUNE_EPOCHS = 5
        print('\n Quick mode: Using reduced epochs for testing')

    # Report Model which will be run
    print('\n Running all tasks for Wildlife classification.')
    print(f'\n Dataset:  {config.DATASET}, on device: {config.DEVICE}')

    # Run Task 1
    try:
        task1_results = train_from_scratch(config, data_fraction=1.0)
        print('Task 1 completed successfully.')
    except Exception as e:
        print(f'Task 1 Failed: {e}')

    # Run Task 2
    try:
        task2_results = fine_tune_pretrained(config, data_fraction=1.0)
        print('Task 2 completed successfully.')
    except Exception as e:
        print(f'Task 2 Failed: {e}')

    # Run Task 3
    try:
        task3_results = zero_shot_classification(config)
        print('Task 3 completed successfully.')
    except Exception as e:
        print(f'Task 3 Failed: {e}')

    # Report Completion of tasks
    print('\n All experiments completed.')
    print(f'\n Results saved in: {config.RESULTS_DIR}')
    print(f' - Models: {config.MODELS_DIR}')
    print(f' - Plots: {config.PLOTS_DIR}')
    print(f' - Metrics: {config.METRICS_DIR}')

if __name__ == '__main__':
    main()
