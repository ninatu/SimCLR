from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config')
    config = parser.parse_args().config

    config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

    simclr = SimCLR(dataset, config)
    simclr.train()


if __name__ == "__main__":
    main()
