# NSQL
Numbers Station Text to SQL model code.

NSQL is a family of autoregressive open-source large foundation models (FMs) designed specifically for SQL generation tasks. All model weights are provided on HuggingFace.

| Model Name | Size | Link |
| ---------- | ---- | ------- |
| NumbersStation/nsql-350M | 350M | [link](https://huggingface.co/NumbersStation/nsql-350M)
| NumbersStation/nsql-2B   | 2B   | [link](https://huggingface.co/NumbersStation/nsql-2B)
| NumbersStation/nsql-6B   | 6B   | [link](https://huggingface.co/NumbersStation/nsql-6B)

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

## Data
Coming soon!