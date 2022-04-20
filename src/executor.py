"""
Module is responsible for:
"""
from argparse import ArgumentParser
import logging
from projects.deep_vasp_e import DeepVaspE
from pathlib import Path
def generate_parser() -> ArgumentParser:
    """
    Generates parser used for CLI

    :return:
    """
    parser = ArgumentParser()
    parser.add_argument('-p', dest="project", default="DeepVaspE")
    return parser

def UIRoutine():
    """

    """


class MainRoutine:
    parser = generate_parser()

    def __init__(self):
        """

        """
        logging.basicConfig(filename='myapp.log', level=logging.INFO)


    @classmethod
    def add_argument(cls, argument):
        cls.parser.add_argument(argument)

    def execute(self):
        """"""

if __name__ == '__main__':
   # UIRoutine()
    p = DeepVaspE(working_directory=Path("C:/Users/felix/PycharmProjects/"))
    p.load_data_sets()
    p.train_model()