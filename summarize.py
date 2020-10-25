from components.Dataset import Dataset
from Summarizer import Summarizer
import torch

import joblib

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)

def backup (e, path):
	joblib.dump(e, path)
	return

def load (path):
	e = joblib.load(path)
	return e

DATA_DIR = "PATH_TO_OUTPUT_TEST_FOLDER" 
TEST_DIR = "PATH_TO_PARSED_TEST_SET" 

# load test set
test_set = load(TEST_DIR)

summy = Summarizer(test_set, "morenolq/SumTO_FNS2020")

summy.summarize(target_dir=DATA_DIR, systemID="YourSystemID", post_editing=True)