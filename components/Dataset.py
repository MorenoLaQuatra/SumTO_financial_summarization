import glob, os, re
from tqdm import tqdm
import nltk
import spacy
import rouge
from joblib import Parallel, delayed

from multiprocess import Process, Manager, Pool

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)

class Dataset():
    def __init__(self, train_dir, test_dir, val_dir, train_dir_gold, test_dir_gold, val_dir_gold):
        self.train_dir = train_dir
        self.test_dir  = test_dir
        self.val_dir   = val_dir
        self.train_dir_gold = train_dir_gold
        self.test_dir_gold  = test_dir_gold
        self.val_dir_gold   = val_dir_gold

        self.train_set = {}
        self.test_set  = {}
        self.val_set   = {}

        self.nlp = spacy.load("en_core_web_sm", disable = ['ner'])
        self.nlp.max_length = 10000000

    def chunks(self, l, n):
        # For item i in a range that is a length of l,
        for i in range(0, len(l), n):
            # Create an index range for l of n items:
            yield l[i:i+n]

    def job_each_file_train(self, k):
        #print (k)
        print(len(self.train_set.keys()), "/", "89898")
        d = {}
        d["key"] = k
        raw_text = open(self.train_dir + k, "r", encoding="utf-8").read()
        d["raw_text"]    = raw_text
        clean_text = self.clean_text(raw_text)
        d["clean_text"]  = clean_text

        self.fill_sentences(d)

        d["summaries"] = []
        for i in [1,2,3,4,5,6,7,8,9,10]:
            try :
                name, extension = k.split(".")
                summary = open(self.train_dir_gold + name + "_" + str(i) + "." + extension, encoding="utf-8").read()
                d["summaries"].append(summary)
            except Exception as e:
                continue
                #print (e)
                #print (self.train_dir_gold + name + "_" + str(i) + "." + extension)

        self.fill_regression_labels(d)
        self.train_set[k] = d

    def parse_train_data(self):
        
        logging.info("Dataset - Parsing Training Data")

        os.chdir(self.train_dir)
        manager = Manager()
        self.train_set = manager.dict()
        
        list_keys = list(glob.glob("*.txt"))
        n_process = 10
        p = Pool(n_process)
        p.map(self.job_each_file_train, list_keys)

        self.train_set = dict(self.train_set)


    def job_each_file_validation(self, k):
        print(len(self.val_set.keys()), "/", "10000")
        d = {}
        d["key"] = k
        d["raw_text"]    = open(self.val_dir + k, "r", encoding="utf-8").read()
        d["clean_text"]  = self.clean_text(d["raw_text"])

        self.fill_sentences(d)

        d["summaries"] = []
        for i in [1,2,3,4,5,6,7,8,9,10]:
            try :
                name, extension = k.split(".")
                summary = open(self.val_dir_gold + name + "_" + str(i) + "." + extension, encoding="utf-8").read()
                d["summaries"].append(summary)
            except Exception as e:
                #print (e)
                #print (self.val_dir_gold + name + "_" + str(i) + "." + extension)
                continue

        self.fill_regression_labels(d)
        self.val_set[k] = d

    def parse_validation_data(self):
        logging.info("Dataset - Parsing Validation Data")

        os.chdir(self.val_dir)
        manager = Manager()
        self.val_set = manager.dict()
        n_process = 10
        list_keys = list(glob.glob("*.txt"))
        p = Pool(n_process)
        p.map(self.job_each_file_validation, list_keys)

        self.val_set = dict(self.val_set)

    def job_each_file_test(self, k):
        print(len(self.test_set.keys()), "/", "10000")
        d = {}
        d["key"] = k
        d["raw_text"]    = open(self.test_dir + k, "r", encoding="utf-8").read()
        d["clean_text"]  = self.clean_text(d["raw_text"])

        self.fill_sentences(d)
        self.test_set[k] = d

    def parse_test_data(self):
        logging.info("Dataset - Parsing Test Data")

        os.chdir(self.test_dir)
        manager = Manager()
        self.test_set = manager.dict()
        n_process = 10
        list_keys = list(glob.glob("*.txt"))
        p = Pool(n_process)
        p.map(self.job_each_file_test, list_keys)

        self.test_set = dict(self.test_set)



    def clean_text(self, text):
        clean = text.replace('\n',' ')
        clean = re.sub(' +', ' ', clean)
        return clean

    def clean_sentence(self, s, remove_punct=True, remove_sym = True, remove_stop=True):
        
        analyzed_sentence = self.nlp(s)
        clean_token = []

        for token in analyzed_sentence:
            if token.pos_ != "PUNCT":
                clean_token.append(token)

        if remove_punct:
            ct = []
            for token in clean_token:
                if token.pos_ != "PUNCT":
                    ct.append(token)
            clean_token = ct

        if remove_sym:
            ct = []
            for token in clean_token:
                if token.is_stop == False:
                    ct.append(token)
            clean_token = ct

        if remove_stop:
            ct = []
            for token in clean_token:
                if token.pos_ != "SYM":
                    ct.append(token)
            clean_token = ct

        return ' '.join(word.text for word in clean_token)

    def fill_sentences(self, d):
        d["raw_sentences"] = {}
        d["clean_sentences"] = {}
        doc = self.nlp(d["clean_text"])
        sentences = doc.sents
        for i, s in enumerate(sentences):
            clean_s = self.clean_sentence(s.text, self.nlp)
            d["raw_sentences"][i] = s.text
            d["clean_sentences"][i] = clean_s

    def fill_regression_labels(self, d):
        d["r2p_labels"] = {}
        d["r2r_labels"] = {}
        d["r2f_labels"] = {}
        d["rlp_labels"] = {}
        d["rlr_labels"] = {}
        d["rlf_labels"] = {}

        for i, s in d["raw_sentences"].items():
            r2p, r2r, r2f, rlp, rlr, rlf = self.get_regression_labels(s, d["summaries"], aggregation="max")
            d["r2p_labels"][i] = r2p
            d["r2r_labels"][i] = r2r
            d["r2f_labels"][i] = r2f
            d["rlp_labels"][i] = rlp
            d["rlr_labels"][i] = rlr
            d["rlf_labels"][i] = rlf

        return d

    def get_regression_labels(self, sentence, list_summaries, aggregation="max"):
        #print ("start get regression")
        r_computer = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], limit_length=False, max_n=2, alpha=0.5, stemming=False)
        r2p = 0
        r2r = 0
        r2f = 0
        rlp = 0
        rlr = 0
        rlf = 0
        if aggregation == "max":
            for summ in list_summaries:
                score = r_computer.get_scores(sentence, summ)
                #print (score)
                if (score["rouge-2"]["p"] > r2p):
                    r2p = score["rouge-2"]["p"]
                if (score["rouge-2"]["r"] > r2r):
                    r2r = score["rouge-2"]["r"]
                if (score["rouge-2"]["f"] > r2f):
                    r2f = score["rouge-2"]["f"]
                
                if (score["rouge-l"]["p"] > rlp):
                    rlp = score["rouge-l"]["p"]
                if (score["rouge-l"]["r"] > rlr):
                    rlr = score["rouge-l"]["r"]
                if (score["rouge-l"]["f"] > rlf):
                    rlf = score["rouge-l"]["f"]
        else:
            print ("Aggregation type: " + aggregation + " not supported yet")
            exit()

        #print ("end get regression")
        
        
        return r2p, r2r, r2f, rlp, rlr, rlf
