"""
Delete an Azure ML Experiment
"""
from azureml.core.experiment import Experiment
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    '--experiment', type=str,
    help='Experiment name to delete.'
)
args = parser.parse_args()

ws = Workspace.from_config(path='./project/.azureml/config.json')
ex = Experiment(ws, args.experiment)
ex.delete()