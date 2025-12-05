from argparse import ArgumentParser
from os import system
from os.path import join
from typing import cast
from omegaconf import OmegaConf, DictConfig

from src.data_generator import process_parallel


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c",
                            "--config",
                            help="Path to YAML configuration file",
                            default="configs/dwk.yaml",
                            type=str)
    return arg_parser


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    config = cast(DictConfig, OmegaConf.load(__args.config))
    joern = config.joern_path
    root = join(config.data_folder, config.dataset.name)
    source_path = join(root, "source-code")
    out_path = join(root, "csv")
    print("config.data_folder:", config.data_folder)
    print("config.dataset.name:", config.dataset.name)
    print("root:", root)
    print("out_path:", out_path)
    print("config.joern_path:", config.joern_path)
    print("source_path:", source_path)
    # try:
    system(f"{joern} {out_path} {source_path}")


