


import os, sys
import yaml
import json

import argparse

this_file_dir = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the finetuning config .yaml file.')
    args = parser.parse_args()

    # load finetuning params
    with open(args.config, 'r') as f:
        finetuning_params = yaml.load(f, Loader=yaml.FullLoader)
    
    path_to_models = os.path.join(finetuning_params['model_version'])



