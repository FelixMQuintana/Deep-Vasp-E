
from pathlib import PurePath

class DeepVaspE:

    def __init__(self, working_directory: PurePath, evaluation_image: PurePath = None):
        self._working_directory = working_directory
        self._evaluation_image = evaluation_image
        for child in self._working_directory: child
            child
