import argparse
from cpca_exp.experiments import run_single_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id")
    args = parser.parse_args()
    run_single_experiment(TASK_ID=int(args.task_id))
