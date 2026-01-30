import argparse
import pandas as pd
import numpy as np


def main(args):
    # TODO: load data
    # TODO: train classifier
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare and consolidate data"
    )
    parser.add_argument("--in_folder", help="the input folder")
    parser.add_argument("--out_folder", help="the output folder")
    args = parser.parse_args()

    main(args)