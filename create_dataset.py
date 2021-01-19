from components.Dataset import Dataset

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

BACKUP_DIR = "<PATH_TO_STORE_PARSED_DATASET>" 
base_dir = "<BASE_PATH_FOR_YOUR_DATASET>" 

data = Dataset( train_dir = base_dir + "train_articles/",
                test_dir  = base_dir + "test_articles/",
                val_dir   = base_dir + "eval_articles/",
                train_dir_gold = base_dir + "train_summaries/",
                test_dir_gold  = base_dir + "test_summaries/",
                val_dir_gold   = base_dir + "eval_summaries/"
                )

data.parse_test_data()
backup(data, BACKUP_DIR + "test_parsed.bkp")
