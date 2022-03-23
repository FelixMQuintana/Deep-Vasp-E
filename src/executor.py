"""
Module is responsible for:
"""
from argparse import ArgumentParser


def generate_parser() -> ArgumentParser:
    """
    Generates parser used for CLI

    :return:
    """
    parser = ArgumentParser()
    parser.add_argument(description='Classify 3D voxel images of protein families')
    # parser.add_argument()
    return parser

# def main(args)
