# SumTO @ FNS 2020
The repository include the evaluation code fot the SumTO summarization system proposed for the FNS 2020 Shared Task. 

## Evaluation script
- `Summarizer.py` include the code for the Summarizer python object. It is able to initialize the model and perform the summarization using the `.summarize()` function
- `summarize.py` contains the code to initialize and apply the model to pre-parsed input data collections.
- In `summarize.py`: `DATA_DIR` and `TEST_DIR` should be set according to your environment configuration
- In `summarize.py`: `YourSystemID` should be set according to your output folder (it will contain the summarized documents at the end of the summarization process)

## Pre-trained Financial model

Available at https://huggingface.co/morenolq/SumTO_FNS2020 or direclty on pytorch using the tag `morenolq/SumTO_FNS2020`


## Citation (Coming Soon)

La Quatra, M., Cagliero L., *End-to-end Training For Financial Report Summarization*

The paper will be available soon.
