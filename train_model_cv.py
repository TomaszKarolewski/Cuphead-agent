"""
_summary_

"""
import os
import random
import copy
import pickle
import numpy as np
import tensorflow as tf


class TrainingAssistant():

    def __init__(self) -> None:
        
        self.project_dir: str = os.path.dirname(os.path.abspath(__file__))
        self.training_set_dir: str = os.path.join(self.project_dir, "Training set")

    def load_input_chunk(self):
        pass

    def build_nn(self):
        pass

    def train_model(self):
        pass
