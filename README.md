# SumTO @ FNS 2020
The repository include the evaluation code fot the SumTO summarization system proposed for the FNS 2020 Shared Task. 

## Evaluation script
- `Summarizer.py` include the code for the Summarizer python object. It is able to initialize the model and perform the summarization using the `.summarize()` function
- `summarize.py` contains the code to initialize and apply the model to pre-parsed input data collections.
- In `summarize.py`: `DATA_DIR` and `TEST_DIR` should be set according to your environment configuration
- In `summarize.py`: `YourSystemID` should be set according to your output folder (it will contain the summarized documents at the end of the summarization process)

## Pre-trained Financial model

Available at https://huggingface.co/morenolq/SumTO_FNS2020 or using the transformers python library with the tag `morenolq/SumTO_FNS2020`


## Citation (Coming Soon)

La Quatra, M., & Cagliero, L. (2020, December). *End-to-end Training For Financial Report Summarization*. In Proceedings of the 1st Joint Workshop on Financial Narrative Processing and MultiLing Financial Summarisation (pp. 118-123).

[https://www.aclweb.org/anthology/2020.fnp-1.20.pdf](https://www.aclweb.org/anthology/2020.fnp-1.20.pdf)
