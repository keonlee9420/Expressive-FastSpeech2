import argparse
import yaml

from preparation import aihub_mmv


def main(args, preprocess_config, model_config):
    if "AIHub-MMV" in preprocess_config["dataset"]:
        aihub_mmv.split_into_dialog(preprocess_config, model_config)
    else:
        raise NotImplementedError("Not supported dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    args = parser.parse_args()

    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)

    main(args, preprocess_config, model_config)
