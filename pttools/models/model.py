"""Template for equations of state"""

import abc

import pttools.type_hints as th


class Model:
    """Template for equations of state"""
    def __init__(self):
        pass

    @abc.abstractmethod
    def p(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def cs2(self, *args, **kwargs):
        pass
