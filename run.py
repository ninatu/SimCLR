from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config')
    config = parser.parse_args().config

    exp_name = os.path.splitext(os.path.basename(config))[0]

    config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(batch_size=config['batch_size'], **config['dataset'])

    simclr = SimCLR(dataset, config, exp_name)
    simclr.train()


if __name__ == "__main__":
    main()
