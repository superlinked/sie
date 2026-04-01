# SIE SDK

Python client SDK for the SIE inference server.

## Installation

```bash
pip install sie-sdk
```

## Quick Start

```python
from sie_sdk import SIEClient
from sie_sdk.types import Item

client = SIEClient("http://localhost:8080")

# Encode text
result = client.encode("BAAI/bge-m3", Item(text="Hello world"))
print(result.dense)  # [0.1, 0.2, ...]

# Score pairs
scores = client.score("cross-encoder", [
    (Item(text="query"), Item(text="document"))
])
```

## License

Apache 2.0
