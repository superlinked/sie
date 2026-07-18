# SIE Python quickstart

This example uses the SIE SDK to create one text embedding. The code is the
same whether SIE runs on your laptop, in your Kubernetes cluster, or as a
managed service.

## Run it

Install the SDK:

```bash
pip install sie-sdk
cd examples/quickstart
```

For a local SIE server on `http://localhost:8080`, no configuration is needed:

```bash
python quickstart.py
```

For self-deployed SIE, point the SDK at your gateway:

```bash
export SIE_CLUSTER_URL="https://sie.example.com"
python quickstart.py
```

For managed SIE, also provide your API key:

```bash
export SIE_CLUSTER_URL="<your SIE endpoint>"
export SIE_API_KEY="<your API key>"
python quickstart.py
```

`SIE_MODEL` optionally selects a different embedding model:

```bash
export SIE_MODEL="BAAI/bge-m3"
```

Keep API keys in environment variables or your normal secrets manager, never
in source files.

The [quickstart notebook](../quickstart.ipynb) continues with scoring and
extraction.

## Test

```bash
PYTHONPATH=. python -m unittest test_quickstart.py
```
