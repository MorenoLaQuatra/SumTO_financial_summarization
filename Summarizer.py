import numpy as np 
import pandas as pd 
import random
import matplotlib.pyplot as plt
import re

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from transformers import BertTokenizer, DistilBertTokenizer
from transformers import BertForSequenceClassification, AdamW, DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import AutoModelForSequenceClassification



from tqdm import tqdm
import os

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)



class Summarizer():
    def __init__(self, dataset, fine_tuned_model, is_validation=False):
        logging.info("Summarizer - initializing summarizer")

        if is_validation:
            self.test_dataset = dataset.val_set
        else:
            self.test_dataset = dataset.test_set

        logging.info("Summarizer - Loading model (auto)")
        self.tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model) #for regression

        self.device = torch.device("cuda")
        free_gpu = int(self.get_freer_gpu())
        logging.info(str(free_gpu))
        logging.info(type(free_gpu))
        torch.cuda.set_device(free_gpu)
        self.device = torch.device('cuda:' + str(free_gpu))

        logging.info("Summarizer - Loading model into device")
        self.model = self.model.to(self.device)

    def get_freer_gpu(self):
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        return np.argmax(memory_available)

    def summarize(self, target_dir, max_len = 1000, systemID="1", post_editing = False):
        for k, p in tqdm(self.test_dataset.items()):
            #create output file
            k_name = k.replace('.txt','')
            fw = open(target_dir + str(k_name) + "_summary-" + systemID + ".txt", "w", encoding="utf-8")

            # encode sentences
            logging.info("Summarizer - Encoding text for doc " + str(k))

            list_text = list(p["raw_sentences"].values())
            input_ids = self.encoding_text(list_text)

            #create dataloader 
            dataloader = TensorDataset(input_ids)
            dataloader = DataLoader(dataloader, batch_size = 64, shuffle = False, num_workers=10)

            scores = []
            # forward pass
            logging.info("Summarizer - Forward Pass")
            for step, batch in enumerate(tqdm(dataloader)):
                outputs = self.model(batch[0].to(self.device))
                list_out = outputs[0].tolist()
                for out in list_out:
                    scores.append(float(out[0]))
                    
                del outputs
                del batch[0]
                torch.cuda.empty_cache()


            #Higher is better
            sorted_ids = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True) 

            summary = ""
            ids_for_summary = []
            str_to_add = ""
            id_to_add = -1

            for i in sorted_ids:
                sent = list_text[i]
                sent = re.sub('[.]{2,}', '', sent)

                if post_editing and (not self.valid_sentence(sent, summary)):
                    continue

                len_sent = len(sent.split())

                if len(summary.split()) + len_sent < max_len:
                    summary += sent + "\n"
                    ids_for_summary.append(i)
                elif len(summary.split()) + len_sent == max_len:
                    summary += sent + "\n"
                    ids_for_summary.append(i)
                    break
                else:
                    len_to_add = max_len - len(summary.split())
                    list_to_add = sent.split()[:len_to_add-1]
                    str_to_add = " ".join([s for s in list_to_add])
                    id_to_add = i
                    break

            # sort ids_for_summary
            ids_for_summary.sort()

            # recreate summary with original order.
            
            summary = ""
            inserted = False
            for i in ids_for_summary:
                if not inserted and id_to_add < i:
                    sent = list_text[id_to_add]
                    sent = re.sub('[.]{2,}', '', sent)
                    summary += sent + "\n"
                    inserted = True
                sent = list_text[i]
                sent = re.sub('[.]{2,}', '', sent)
                summary += sent + "\n"

            summary = summary.strip()
            fw.write(summary)
            fw.close()



    def valid_sentence(self, sent, summary, threshold_sym = 0.50, threshold_upper = 0.50, threshold_min_length=5):
        # return True if a good sentence, False if it should be skipped
        
        if summary.find(sent) != -1:
            logging.info("Summarizer - skipping sentence **" + sent + "** already in summary")
            return False

        if len(sent.split(" ")) < threshold_min_length:
            logging.info("Summarizer - skipping sentence **" + sent + "** too short -> less than " + str(threshold_min_length))
            return False


        n_sym = 0
        for char in sent:
            if not(char.isalpha()):
                n_sym += 1

        perc_of_sym = n_sym / len(sent)
        if perc_of_sym > threshold_sym:
            logging.info("Summarizer - skipping sentence **" + sent + "** too much symbols")
            return False

        n_upper = 0
        for char in sent:
            if char.isupper():
                n_upper += 1

        perc_of_upper = n_upper / len(sent)
        if perc_of_upper > threshold_upper:
            logging.info("Summarizer - skipping sentence **" + sent + "** too much uppercase chars")
            return False
        
        return True


    def encoding_text (self, text_list):
        logging.info("Summarizer - encoding user-defined data")
        all_input_ids = []    
        for text in tqdm(text_list):
            input_ids = self.tokenizer.encode(
                            text,                      
                            add_special_tokens = True, 
                            max_length = 256,           
                            pad_to_max_length = True,
                            return_tensors = 'pt'  
                    )
            all_input_ids.append(input_ids)    
        all_input_ids = torch.cat(all_input_ids, dim=0)
        return all_input_ids
