# NLGorgias

This project provide POC for a system between NL and [Gorgias Cloud](http://gorgiasb.tuc.gr/GorgiasCloud.html).

## Structure

`paper_dataset` folder contains the dataset used during the whole project, the `data/dataset_10K.json` was the next step but due to a lack of time we couldn't do it.

In the `src` folder we also have the `paper_code` folders that contains codes used during the project but couldn't be converted to a production grade code for the same reason as above.

However the `dataset_generation.py`, `split_format_dataset.py`, `finetune_llm.py` are good enough to be used.

## Configuration
For the `.env` file :
```
USER=XYZ
PASSWORD=XXXX
HF_TOKEN=XXXXX
OPENAI_API_KEY=XXXX
```
USER and PASSWORD are for the Gorgias Cloud service, HF for HuggingFace and OpenAI for ChatGPT.