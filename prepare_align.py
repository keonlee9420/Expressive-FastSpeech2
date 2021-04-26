import argparse

import yaml

from preprocessor import aihub_mmv


def main(config):
    if "AIHub-MMV" in config["dataset"]:
        aihub_mmv.prepare_align(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)
