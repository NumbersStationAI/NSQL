## Data Preprocessing

We provide scripts and instructions to create the [NSText2SQL](https://huggingface.co/datasets/NumbersStation/NSText2SQL) to train [NSQL](https://huggingface.co/NumbersStation/nsql-6B) models. The dataset will saved as jsonl files, following the format:

```
{"instruction": ..., "output": "...", "source": "..."}
```

#### Data Download

We use the datasets hosted on Github, Huggingface, and other online servers. You will need to download the datasets by running the following commands from the `data` folder:

```bash
cd data/
bash download.sh
cd ..
```

To download spider dataset, you need to download it from [here](https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ) and unzip it to the data folder.

#### Data Preparation

To preprocess the data into our own format. You can run:

```bash
python prep_text2sql_data.py \
  --datasets academic \
  --datasets advising \
  --output_dir [OUTPUT_DIR]
```

The processed data will be saved into `[OUPUT_DIR]/YYYY-MM-DD` folder. Here is the available DATASET_NAME list:
- wikisql
- academic
- advising
- atis
- imdb
- restaurants
- scholar
- yelp
- sede
- eicu
- mimic_iii
- GeoNuclearData
- GreaterManchesterCrime
- Pesticide
- StudentMathScore
- TheHistoryofBaseball
- USWildFires
- WhatCDHipHop
- WorldSoccerDataBase
- mimicsql_data
- criteria2sql
- sql_create_context
- squall
- css
- spider
- nvbench

For more information you can find in the `prep_text2sql_data.py`.