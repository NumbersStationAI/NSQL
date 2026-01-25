# NSQL
Numbers Station Text to SQL model code.

NSQL is a family of autoregressive open-source large foundation models (FMs) designed specifically for SQL generation tasks. All model weights are provided on HuggingFace.

| Model Name | Size | Link |
| ---------- | ---- | ------- |
| NumbersStation/nsql-350M       | 350M | [link](https://huggingface.co/NumbersStation/nsql-350M)
| NumbersStation/nsql-2B         | 2.7B | [link](https://huggingface.co/NumbersStation/nsql-2B)
| NumbersStation/nsql-6B         | 6B   | [link](https://huggingface.co/NumbersStation/nsql-6B)
| NumbersStation/nsql-llama-2-7B | 7B   | [link](https://huggingface.co/NumbersStation/nsql-llama-2-7B)

## Setup
To install, run
```
pip install -r requirements.txt
```

## Usage
See examples in `examples/` for how to connect to Postgres or SQLite to ask questions directly over your data. A small code snippet is provided below from the `examples/` directory.

In a separate screen or window, run
```bash
python3 -m manifest.api.app \
    --model_type huggingface \
    --model_generation_type text-generation \
    --model_name_or_path NumbersStation/nsql-350M \
    --device 0
```

Then run

```python
from db_connectors import PostgresConnector
from prompt_formatters import RajkumarFormatter
from manifest import Manifest

postgres_connector = PostgresConnector(
    user=USER, password=PASSWORD, dbname=DATABASE, host=HOST, port=PORT
)
postgres_connector.connect()
db_schema = [postgres_connector.get_schema(table) for table in postgres_connector.get_tables()]
formatter = RajkumarFormatter(db_schema)

manifest_client = Manifest(client_name="huggingface", client_connection="http://127.0.0.1:5000")

def get_sql(instruction: str, max_tokens: int = 300) -> str:
    prompt = formatter.format_prompt(instruction)
    res = manifest_client.run(prompt, max_tokens=max_tokens)
    return formatter.format_model_output(res)

print(get_sql("Number of rows in table?"))
```

## Data Preparation

In `data_prep` folder, we provide data preparation scripts to generate [NSText2SQL](https://huggingface.co/datasets/NumbersStation/NSText2SQL) to train [NSQL](https://huggingface.co/NumbersStation/nsql-6B) models.

## License

The code in this repo is licensed under the Apache 2.0 license. Unless otherwise noted,

```
Copyright 2023 Numbers Station

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

The data to generate NSText2SQL is sourced from repositories with various licenses. Any use of all or part of the data gathered in NSText2SQL must abide by the terms of the original licenses, including attribution clauses when relevant. We thank all authors who provided these datasets. We provide provenance information for each dataset below.

| Datasets               | License      | Link                                                                                                                 |
| ---------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------- |
| academic               | Not Found    | [https://github.com/jkkummerfeld/text2sql-data](https://github.com/jkkummerfeld/text2sql-data)                       |
| advising               | CC-BY-4.0    | [https://github.com/jkkummerfeld/text2sql-data](https://github.com/jkkummerfeld/text2sql-data)                       |
| atis                   | Not Found    | [https://github.com/jkkummerfeld/text2sql-data](https://github.com/jkkummerfeld/text2sql-data)                       |
| restaurants            | Not Found    | [https://github.com/jkkummerfeld/text2sql-data](https://github.com/jkkummerfeld/text2sql-data)                       |
| scholar                | Not Found    | [https://github.com/jkkummerfeld/text2sql-data](https://github.com/jkkummerfeld/text2sql-data)                       |
| imdb                   | Not Found    | [https://github.com/jkkummerfeld/text2sql-data](https://github.com/jkkummerfeld/text2sql-data)                       |
| yelp                   | Not Found    | [https://github.com/jkkummerfeld/text2sql-data](https://github.com/jkkummerfeld/text2sql-data)                       |
| criteria2sql           | Apache-2.0   | [https://github.com/xiaojingyu92/Criteria2SQL](https://github.com/xiaojingyu92/Criteria2SQL)                         |
| css                    | CC-BY-4.0    | [https://huggingface.co/datasets/zhanghanchong/css](https://huggingface.co/datasets/zhanghanchong/css)               |
| eICU                   | CC-BY-4.0    | [https://github.com/glee4810/EHRSQL](https://github.com/glee4810/EHRSQL)                                             |
| mimic_iii              | CC-BY-4.0    | [https://github.com/glee4810/EHRSQL](https://github.com/glee4810/EHRSQL)                                             |
| geonucleardata         | CC-BY-SA-4.0 | [https://github.com/chiahsuan156/KaggleDBQA](https://github.com/chiahsuan156/KaggleDBQA)                             |
| greatermanchestercrime | CC-BY-SA-4.0 | [https://github.com/chiahsuan156/KaggleDBQA](https://github.com/chiahsuan156/KaggleDBQA)                             |
| studentmathscore       | CC-BY-SA-4.0 | [https://github.com/chiahsuan156/KaggleDBQA](https://github.com/chiahsuan156/KaggleDBQA)                             |
| thehistoryofbaseball   | CC-BY-SA-4.0 | [https://github.com/chiahsuan156/KaggleDBQA](https://github.com/chiahsuan156/KaggleDBQA)                             |
| uswildfires            | CC-BY-SA-4.0 | [https://github.com/chiahsuan156/KaggleDBQA](https://github.com/chiahsuan156/KaggleDBQA)                             |
| whatcdhiphop           | CC-BY-SA-4.0 | [https://github.com/chiahsuan156/KaggleDBQA](https://github.com/chiahsuan156/KaggleDBQA)                             |
| worldsoccerdatabase    | CC-BY-SA-4.0 | [https://github.com/chiahsuan156/KaggleDBQA](https://github.com/chiahsuan156/KaggleDBQA)                             |
| pesticide              | CC-BY-SA-4.0 | [https://github.com/chiahsuan156/KaggleDBQA](https://github.com/chiahsuan156/KaggleDBQA)                             |
| mimicsql_data          | MIT          | [https://github.com/wangpinggl/TREQS](https://github.com/wangpinggl/TREQS)                                           |
| nvbench                | MIT          | [https://github.com/TsinghuaDatabaseGroup/nvBench](https://github.com/TsinghuaDatabaseGroup/nvBench)                 |
| sede                   | Apache-2.0   | [https://github.com/hirupert/sede](https://github.com/hirupert/sede)                                                 |
| spider                 | CC-BY-SA-4.0 | [https://huggingface.co/datasets/spider](https://huggingface.co/datasets/spider)                                     |
| sql_create_context     | CC-BY-4.0    | [https://huggingface.co/datasets/b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) |
| squall                 | CC-BY-SA-4.0 | [https://github.com/tzshi/squall](https://github.com/tzshi/squall)                                                   |
| wikisql                | BSD 3-Clause | [https://github.com/salesforce/WikiSQL](https://github.com/salesforce/WikiSQL)                                       |

For full terms, see the LICENSE file. If you have any questions, comments, or concerns about licensing please [contact us](https://www.numbersstation.ai/signup).

# Citing this work

If you use this data in your work, please cite our work _and_ the appropriate original sources:

To cite NSText2SQL, please use:
```TeX
@software{numbersstation2023NSText2SQL,
  author    = {Numbers Station Labs},
  title     = {NSText2SQL: An Open Source Text-to-SQL Dataset for Foundation Model Training},
  month     = {July},
  year      = {2023},
  url       = {https://github.com/NumbersStationAI/NSQL},
}
```

To cite dataset used in this work, please use:

| Datasets               | Cite                                                                                     |
| ---------------------- | ---------------------------------------------------------------------------------------- |
| academic               | `\cite{data-advising,data-academic}`                                                     |
| advising               | `\cite{data-advising}`                                                                   |
| atis                   | `\cite{data-advising,data-atis-original,data-atis-geography-scholar}`                    |
| restaurants            | `\cite{data-advising,data-restaurants-logic,data-restaurants-original,data-restaurants}` |
| scholar                | `\cite{data-advising,data-atis-geography-scholar}`                                       |
| imdb                   | `\cite{data-advising,data-imdb-yelp}`                                                    |
| yelp                   | `\cite{data-advising,data-imdb-yelp}`                                                    |
| criteria2sql           | `\cite{Criteria-to-SQL}`                                                                 |
| css                    | `\cite{zhang2023css}`                                                                    |
| eICU                   | `\cite{lee2022ehrsql}`                                                                   |
| mimic_iii              | `\cite{lee2022ehrsql}`                                                                   |
| geonucleardata         | `\cite{lee-2021-kaggle-dbqa}`                                                            |
| greatermanchestercrime | `\cite{lee-2021-kaggle-dbqa}`                                                            |
| studentmathscore       | `\cite{lee-2021-kaggle-dbqa}`                                                            |
| thehistoryofbaseball   | `\cite{lee-2021-kaggle-dbqa}`                                                            |
| uswildfires            | `\cite{lee-2021-kaggle-dbqa}`                                                            |
| whatcdhiphop           | `\cite{lee-2021-kaggle-dbqa}`                                                            |
| worldsoccerdatabase    | `\cite{lee-2021-kaggle-dbqa}`                                                            |
| pesticide              | `\cite{lee-2021-kaggle-dbqa}`                                                            |
| mimicsql_data          | `\cite{wang2020text}`                                                                    |
| nvbench                | `\cite{nvBench_SIGMOD21}`                                                                |
| sede                   | `\cite{hazoom2021text}`                                                                  |
| spider                 | `\cite{data-spider}`                                                                     |
| sql_create_context     | Not Found                                                                                |
| squall                 | `\cite{squall}`                                                                          |
| wikisql                | `\cite{data-wikisql}`                                                                    |


```TeX
@InProceedings{data-advising,
  dataset   = {Advising},
  author    = {Catherine Finegan-Dollak, Jonathan K. Kummerfeld, Li Zhang, Karthik Ramanathan, Sesh Sadasivam, Rui Zhang, and Dragomir Radev},
  title     = {Improving Text-to-SQL Evaluation Methodology},
  booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month     = {July},
  year      = {2018},
  location  = {Melbourne, Victoria, Australia},
  pages     = {351--360},
  url       = {http://aclweb.org/anthology/P18-1033},
}

@InProceedings{data-imdb-yelp,
  dataset   = {IMDB and Yelp},
  author    = {Navid Yaghmazadeh, Yuepeng Wang, Isil Dillig, and Thomas Dillig},
  title     = {SQLizer: Query Synthesis from Natural Language},
  booktitle = {International Conference on Object-Oriented Programming, Systems, Languages, and Applications, ACM},
  month     = {October},
  year      = {2017},
  pages     = {63:1--63:26},
  url       = {http://doi.org/10.1145/3133887},
}

@article{data-academic,
  dataset   = {Academic},
  author    = {Fei Li and H. V. Jagadish},
  title     = {Constructing an Interactive Natural Language Interface for Relational Databases},
  journal   = {Proceedings of the VLDB Endowment},
  volume    = {8},
  number    = {1},
  month     = {September},
  year      = {2014},
  pages     = {73--84},
  url       = {http://dx.doi.org/10.14778/2735461.2735468},
} 

@InProceedings{data-atis-geography-scholar,
  dataset   = {Scholar, and Updated ATIS and Geography},
  author    = {Srinivasan Iyer, Ioannis Konstas, Alvin Cheung, Jayant Krishnamurthy, and Luke Zettlemoyer},
  title     = {Learning a Neural Semantic Parser from User Feedback},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year      = {2017},
  pages     = {963--973},
  location  = {Vancouver, Canada},
  url       = {http://www.aclweb.org/anthology/P17-1089},
}

@article{data-atis-original,
  dataset   = {ATIS, original},
  author    = {Deborah A. Dahl, Madeleine Bates, Michael Brown, William Fisher, Kate Hunicke-Smith, David Pallett, Christine Pao, Alexander Rudnicky, and Elizabeth Shriber},
  title     = {{Expanding the scope of the ATIS task: The ATIS-3 corpus}},
  journal   = {Proceedings of the workshop on Human Language Technology},
  year      = {1994},
  pages     = {43--48},
  url       = {http://dl.acm.org/citation.cfm?id=1075823},
}

@inproceedings{data-restaurants-logic,
  author    = {Lappoon R. Tang and Raymond J. Mooney},
  title     = {Automated Construction of Database Interfaces: Intergrating Statistical and Relational Learning for Semantic Parsing},
  booktitle = {2000 Joint SIGDAT Conference on Empirical Methods in Natural Language Processing and Very Large Corpora},
  year      = {2000},
  pages     = {133--141},
  location  = {Hong Kong, China},
  url       = {http://www.aclweb.org/anthology/W00-1317},
}

@inproceedings{data-restaurants-original,
 author    = {Ana-Maria Popescu, Oren Etzioni, and Henry Kautz},
 title     = {Towards a Theory of Natural Language Interfaces to Databases},
 booktitle = {Proceedings of the 8th International Conference on Intelligent User Interfaces},
 year      = {2003},
 location  = {Miami, Florida, USA},
 pages     = {149--157},
 url       = {http://doi.acm.org/10.1145/604045.604070},
}

@inproceedings{data-restaurants,
  author    = {Alessandra Giordani and Alessandro Moschitti},
  title     = {Automatic Generation and Reranking of SQL-derived Answers to NL Questions},
  booktitle = {Proceedings of the Second International Conference on Trustworthy Eternal Systems via Evolving Software, Data and Knowledge},
  year      = {2012},
  location  = {Montpellier, France},
  pages     = {59--76},
  url       = {https://doi.org/10.1007/978-3-642-45260-4_5},
}

@InProceedings{data-spider,
  author    = {Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir Radev},
  title     = {Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  year      = {2018},
  location  = {Brussels, Belgium},
  pages     = {3911--3921},
  url       = {http://aclweb.org/anthology/D18-1425},
}

@article{data-wikisql,
  author    = {Victor Zhong, Caiming Xiong, and Richard Socher},
  title     = {Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning},
  year      = {2017},
  journal   = {CoRR},
  volume    = {abs/1709.00103},
}

@InProceedings{Criteria-to-SQL,
  author    = {Yu, Xiaojing  and  Chen, Tianlong  and  Yu, Zhengjie  and  Li, Huiyu  and  Yang, Yang  and  Jiang, Xiaoqian  and  Jiang, Anxiao},
  title     = {Dataset and Enhanced Model for Eligibility Criteria-to-SQL Semantic Parsing},
  booktitle = {Proceedings of The 12th Language Resources and Evaluation Conference},
  month     = {May},
  year      = {2020},
  address   = {Marseille, France},
  publisher = {European Language Resources Association},
  pages     = {5831--5839},
}

@misc{zhang2023css,
  title     = {CSS: A Large-scale Cross-schema Chinese Text-to-SQL Medical Dataset}, 
  author    = {Hanchong Zhang and Jieyu Li and Lu Chen and Ruisheng Cao and Yunyan Zhang and Yu Huang and Yefeng Zheng and Kai Yu},
  year      = {2023},
}

@article{lee2022ehrsql,
  title     = {EHRSQL: A Practical Text-to-SQL Benchmark for Electronic Health Records},
  author    = {Lee, Gyubok and Hwang, Hyeonji and Bae, Seongsu and Kwon, Yeonsu and Shin, Woncheol and Yang, Seongjun and Seo, Minjoon and Kim, Jong-Yeup and Choi, Edward},
  journal   = {Advances in Neural Information Processing Systems},
  volume    = {35},
  pages     = {15589--15601},
  year      = {2022},
}

@inproceedings{lee-2021-kaggle-dbqa,
  title     = {KaggleDBQA: Realistic Evaluation of Text-to-SQL Parsers},
  author    = {Lee, Chia-Hsuan and Polozov, Oleksandr and Richardson, Matthew},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  pages     = {2261--2273},
  year      = {2021},
}

@inproceedings{squall,
  title     = {On the Potential of Lexico-logical Alignments for Semantic Parsing to {SQL} Queries},
  author    = {Tianze Shi and Chen Zhao and Jordan Boyd-Graber and Hal {Daum\'{e} III} and Lillian Lee},
  booktitle = {Findings of EMNLP},
  year      = {2020},
}

@article{hazoom2021text,
  title     = {Text-to-SQL in the wild: a naturally-occurring dataset based on Stack exchange data},
  author    = {Hazoom, Moshe and Malik, Vibhor and Bogin, Ben},
  journal   = {arXiv preprint arXiv:2106.05006},
  year      = {2021},
}

@inproceedings{wang2020text,
  title     = {Text-to-SQL Generation for Question Answering on Electronic Medical Records},
  author    = {Wang, Ping and Shi, Tian and Reddy, Chandan K},
  booktitle = {Proceedings of The Web Conference 2020},
  pages     = {350--361},
  year      = {2020},
}

@inproceedings{nvBench_SIGMOD21,
  title     = {Synthesizing Natural Language to Visualization (NL2VIS) Benchmarks from NL2SQL Benchmarks},
  author    = {Yuyu Luo and Nan Tang and Guoliang Li and Chengliang Chai and Wenbo Li and Xuedi Qin},
  booktitle = {Proceedings of the 2021 International Conference on Management of Data, {SIGMOD} Conference 2021, June 20–25, 2021, Virtual Event, China},
  publisher = {ACM},
  year      = {2021},
}
```

## Benchmark Results

### Spider Benchmark Performance

NSQL-llama-2-7B achieves near-parity with GPT-4 on overall execution accuracy while being **~250× smaller**, and **significantly outperforms GPT-4 on complex queries**.

#### Overall Results

| Model | Parameters | Execution Acc | Matching Acc |
|-------|------------|---------------|--------------|
| **NSQL-llama-2-7B** | **7B** | **75.0%** | **66.3%** |
| GPT-4 | ~1.8T | 76.2% | 41.9% |
| GPT-3.5 | — | 72.8% | 44.2% |
| Llama-2-7B (base) | 7B | 29.1% | 19.3% |

#### Performance by Query Complexity

| Query Type | NSQL-7B | GPT-4 | Δ |
|------------|---------|-------|---|
| **Join** | **53.7%** | ~37.6% | **+43%** |
| **Nested** | **57.2%** | ~37.1% | **+54%** |
| Simple | 91.4% | Higher | — |

**Key insight:** NSQL outperforms GPT-4 on complex queries (joins, nested) while achieving near-parity overall at 250× smaller size.

### Why This Matters

- ✅ **+43% better on Join queries** — critical for multi-table enterprise data
- ✅ **+54% better on Nested queries** — complex business logic
- ✅ **+24.4 points on Matching Accuracy** — structurally correct SQL
- ✅ **250× smaller** — runs locally, no API calls, complete privacy
- ✅ **Near-parity overall** — 75.0% vs 76.2% execution accuracy

### Full Model Comparison

<details>
<summary>Click to expand full benchmark table</summary>

#### Open Models

| Model | Size | Spider Exec | Spider Match | GeoQuery Exec | GeoQuery Match |
|-------|------|-------------|--------------|---------------|----------------|
| NSQL-350M | 350M | 51.7% | 45.6% | 18.4% | 4.3% |
| NSQL-2B | 2B | 59.3% | 53.2% | 18.4% | 15.2% |
| NSQL-6B | 6B | 63.6% | 57.4% | 26.5% | 15.2% |
| **NSQL-llama-2-7B** | **7B** | **75.0%** | **66.3%** | **26.5%** | **30.4%** |
| Llama-2-7B | 7B | 29.1% | 19.3% | 0% | 8.7% |
| Llama-2-13B | 13B | 44.9% | 32.2% | 12.2% | 26.1% |
| Llama-2-70B | 70B | 61.5% | 35.4% | 12.2% | 28.3% |
| StarCoder | 15.5B | 57.7% | 36.0% | 12.2% | 13.0% |

#### Closed Models

| Model | Size | Spider Exec | Spider Match | GeoQuery Exec | GeoQuery Match |
|-------|------|-------------|--------------|---------------|----------------|
| Text-Davinci-003 | 175B | 72.0% | 49.9% | 24.5% | 41.3% |
| GPT-3.5 Chat | — | 72.8% | 44.2% | 46.9% | 39.1% |
| **GPT-4** | **~1.8T** | **76.2%** | **41.9%** | **55.1%** | **39.1%** |

</details>

### Query Type Breakdown (NSQL Models)

| Model | Join | Nested | Simple |
|-------|------|--------|--------|
| NSQL-6B | 37.5% | 37.1% | 83.5% |
| **NSQL-llama-2-7B** | **53.7%** | **57.2%** | **91.4%** |

---

## Evaluation

**Benchmark:** Spider (Yu et al., 2018) — the standard academic evaluation for Text-to-SQL

**Metrics:**
- **Execution Accuracy:** Does the SQL return correct results?
- **Matching Accuracy:** Does the SQL structure match ground truth?

**Query Categories:**
- **Join:** Queries requiring multiple table joins
- **Nested:** Queries with subqueries
- **Simple:** Single-table queries

---


## Contributors

- **Vishal Motwani** — Founding Product Engineer, Numbers Station AI  
- **Sen Wu** — Co-founder, Numbers Station AI  
- **Laurel Orr** — Principal Developer, Numbers Station AI

---

## Citation

If you use NSQL in your research, please cite:

```bibtex
@misc{nsql2024,
  title={NSQL: An Open-Source Text-to-SQL Model Family},
  author={Numbers Station AI},
  year={2024},
  url={https://huggingface.co/NumbersStation}
}
```


## Acknowledgement
We are appreciative to the work done by the all authors for those datasets that made this project possible.