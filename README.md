<!-- ── HERO block ── -->
<div align="center">

<!-- Logo (auto dark/light) -->
<picture>
  <source srcset="https://cdn.prod.website-files.com/65dce6831bf9f730421e2915/66ef0317ed8616151ee1d451_superlinked_logo_white.png"
          media="(prefers-color-scheme: dark)">
  <img width="320"
       src="https://cdn.prod.website-files.com/65dce6831bf9f730421e2915/65dce6831bf9f730421e2929_superlinked_logo.svg"
       alt="Superlinked logo">
</picture>

<!-- Primary CTA -->
<p>
  <a href="https://links.superlinked.com/deep_wiki_repo" style="text-decoration:none;display:inline-block">
    <img src="https://img.shields.io/badge/chat%20with%20our%20AI%20docs-%E2%86%92-72A1FF?style=for-the-badge&logo=readthedocs&logoColor=white"
         alt="chat with our AI docs" width="200">
  </a>
</p>

<!-- Secondary badges -->
<p style="margin:6px 0;text-decoration:none">
  <a href="https://docs.superlinked.com/"                     style="text-decoration:none;display:inline-block"><img src="https://img.shields.io/badge/Docs-orange?logo=readthedocs"                                        alt="Docs"></a>
  <a href="https://deepwiki.com/superlinked/superlinked"      style="text-decoration:none;display:inline-block"><img src="https://deepwiki.com/badge.svg"                                                       alt="DeepWiki"></a>
  <a href="https://pypi.org/project/superlinked/"             style="text-decoration:none;display:inline-block"><img src="https://img.shields.io/pypi/v/superlinked"                                             alt="PyPI"></a>
  <a href="https://github.com/superlinked/superlinked/stargazers" style="text-decoration:none;display:inline-block"><img src="https://img.shields.io/github/stars/superlinked/superlinked?style=social" alt="Stars"></a>
  <img src="https://img.shields.io/github/last-commit/superlinked/superlinked" style="display:inline-block" alt="Last commit">
  <img src="https://img.shields.io/github/license/superlinked/superlinked"     style="display:inline-block" alt="License">
</p>

</div>

<p align="center">
  <em>Build high-performance <b> AI search </b> applications for both humans and agents. </em>
</p>

## Quickstart
```python
%pip install superlinked
```
```python
from superlinked import framework as sl  
  
# Define schema for movie reviews  
class Review(sl.Schema):  
    id: sl.IdField  
    text: sl.String  
  
review = Review()  
space = sl.TextSimilaritySpace(text=review.text, model="all-MiniLM-L6-v2")  
index = sl.Index(space)  
query = sl.Query(index).find(review).similar(space, sl.Param("search")).select_all()  
  
# Setup and run  
source = sl.InMemorySource(review)  
app = sl.InMemoryExecutor(sources=[source], indices=[index]).run()  
  
# Add data and search  
source.put([  
    {"id": "1", "text": "Amazing acting and great story"},  
    {"id": "2", "text": "Boring plot with bad acting"}  
])  
  
result = app.query(query, search="excellent performance")  
print(sl.PandasConverter.to_pandas(result))

```

<details>
<summary><strong>Table&nbsp;of&nbsp;Contents</strong></summary>

- [Table of Contents](#table-of-contents)
- [Quickstart](#quickstart)
- [Overview](#overview)
- [Hands on tutorials](#hands-on-tutorials)
- [Use-cases](#use-cases)
- [Experiment in a notebook](#experiment-in-a-notebook)
    - [Install the superlinked library](#install-the-superlinked-library)
    - [Run the example:](#run-the-example)
- [Run in production](#run-in-production)
  - [Supported Vector Databases](#supported-vector-databases)
- [Logging](#logging)
- [Resources](#resources)
- [Support](#support)

</details>

## Overview

- **WHY**: Improve your vector search relevance by encoding metadata together with your unstructured data into vectors.
- **WHAT**: A framework and a self-hostable REST API server that connects your data, vector database and backend services.
- **HOW**: Construct custom data & query embedding models from pre-trained encoders from `sentence-transformers`, `open-clip` and custom encoders for numbers, timestamps and categorical data. See the [tutorials](#hands-on-tutorials) and [use-case](#use-cases) notebooks below for examples.

If you like what we do, give us a star! ⭐

![](./docs/images/sl_diagram.png)


## Hands-on Tutorials

| Level | What you’ll build & learn | Try it now |
|-------|---------------------------|------------|
| **Start here** | Embed text · images · numbers · categories · time · events. | Text embedding&nbsp;<a href="https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/feature/text_embedding.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a><br>Image embedding&nbsp;<a href="https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/feature/image_embedding.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a><br>Numeric (min-max)&nbsp;<a href="https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/feature/number_embedding_minmax.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a><br>Numeric (similar)&nbsp;<a href="https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/feature/number_embedding_similar.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a><br>Categorical&nbsp;<a href="https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/feature/categorical_embedding.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a><br>Recency embedding&nbsp;<a href="https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/feature/recency_embedding.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a><br>Event effects&nbsp;<a href="https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/feature/event_effects.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a> |
| **Build & extend** | Combine spaces or add custom / optional schemas. | Combine embeddings&nbsp;<a href="https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/feature/combine_multiple_embeddings.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a><br>Custom space&nbsp;<a href="https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/feature/custom_space.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a><br>Optional fields&nbsp;<a href="https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/feature/optional_schema_fields.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a> |
| **Optimise relevance** | Real-time updates & query-time personalisation. | Dynamic parameters&nbsp;<a href="https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/feature/dynamic_parameters.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a><br>Query-time weights&nbsp;<a href="https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/feature/query_time_weights.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a><br>Query result&nbsp;<a href="https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/feature/query_result.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a> |
| **Search & filter** | NL search + hard filters. | NL query&nbsp;<a href="https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/feature/natural_language_querying.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a><br>Hard filtering&nbsp;<a href="https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/feature/hard_filtering.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a><br>Query options&nbsp;<a href="https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/feature/querying_options.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a><br>Vector parts&nbsp;<a href="https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/feature/accessing_stored_vector_parts.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a> |
| **Analyse & export** | Sample embeddings offline. | Vector sampler&nbsp;<a href="https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/feature/vector_sampler.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a> |
| **Go multi-modal** | Unified text + image space. | Multimodal search&nbsp;<a href="https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/feature/image_embedding.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a> |


> 💡 **Want even more?** Browse the complete list of features & concepts in our docs →  
> <https://docs.superlinked.com/concepts/overview>


## Experiment in a notebook

Let's build an e-commerce product search that understands product descriptions and ratings:

#### Run the notebook example:

>First run will take a minute to download the embedding model.  

```python
%pip install superlinked
```
```python
import json
import os

from superlinked import framework as sl


class Product(sl.Schema):
    id: sl.IdField
    description: sl.String
    rating: sl.Integer


product = Product()

description_space = sl.TextSimilaritySpace(
    text=product.description, model="Alibaba-NLP/gte-large-en-v1.5"
)
rating_space = sl.NumberSpace(
    number=product.rating, min_value=1, max_value=5, mode=sl.Mode.MAXIMUM
)
index = sl.Index([description_space, rating_space], fields=[product.rating])


# Define your query and parameters to set them directly at query-time
# or let an LLM fill them in for you using the `natural_language_query` param.
# Don't forget to set your OpenAI API key to unlock this feature.
query = (
    sl.Query(
        index,
        weights={
            description_space: sl.Param("description_weight"),
            rating_space: sl.Param("rating_weight"),
        },
    )
    .find(product)
    .similar(
        description_space,
        sl.Param(
            "description_query",
            description="The text in the user's query that refers to product descriptions.",
        ),
    )
    .limit(sl.Param("limit"))
    .with_natural_query(
        sl.Param("natural_language_query"),
        sl.OpenAIClientConfig(api_key=os.environ["OPEN_AI_API_KEY"], model="gpt-4o")
    )
)

# Run the app in-memory (server & Apache Spark executors available too!).
source = sl.InMemorySource(product)
executor = sl.InMemoryExecutor(sources=[source], indices=[index])
app = executor.run()


# Ingest data into the system - index updates and other processing happens automatically.
source.put([
    {
        "id": 1,
        "description": "Budget toothbrush in black color. Just what you need.",
        "rating": 1,
    },
    {
        "id": 2,
        "description": "High-end toothbrush created with no compromises.",
        "rating": 5,
    },
    {
        "id": 3,
        "description": "A toothbrush created for the smart 21st century man.",
        "rating": 3,
    },
])

result = app.query(query, natural_query="best toothbrushes", limit=1)

# Examine the extracted parameters from your query
print(json.dumps(result.metadata, indent=2))

# The result is the 5-star rated product.
sl.PandasConverter.to_pandas(result)
```

### Using MiniMax as LLM and Embedding Provider

Superlinked supports [MiniMax](https://www.minimax.io/) as both an LLM provider for natural language queries and an embedding provider for text similarity.

**Natural Language Querying with MiniMax:**

```python
import os
from superlinked import framework as sl

# Use MiniMax for natural language query parameter extraction
query = (
    sl.Query(index, weights={...})
    .find(product)
    .similar(description_space, sl.Param("description_query"))
    .limit(sl.Param("limit"))
    .with_natural_query(
        sl.Param("natural_language_query"),
        sl.MiniMaxClientConfig(
            api_key=os.environ["MINIMAX_API_KEY"],
            model="MiniMax-M2.7",  # or "MiniMax-M2.5", "MiniMax-M2.5-highspeed"
        ),
    )
)
```

**Text Embeddings with MiniMax embo-01:**

```python
import os
from superlinked import framework as sl

# Use MiniMax embo-01 for text embeddings (1536 dimensions)
space = sl.TextSimilaritySpace(
    text=review.text,
    model="embo-01",
    model_handler=sl.TextModelHandler.MINIMAX,
    embedding_engine_config=sl.MiniMaxEngineConfig(
        api_key=os.environ["MINIMAX_API_KEY"],
    ),
)
```

## Use-cases

Dive deeper with our notebooks into how each use-case benefits from the Superlinked framework.

- **RAG**: [HR Knowledgebase](https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/rag_hr_knowledgebase.ipynb)
- **Semantic Search**: [Movies](https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/semantic_search_netflix_titles.ipynb), [Business News](https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/semantic_search_news.ipynb), [Product Images & Descriptions](https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/image_search_e_commerce.ipynb)
- **Recommendation Systems**: [E-commerce](https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/recommendations_e_commerce.ipynb)
- **Analytics**: [User Acquisition](https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/analytics_user_acquisition.ipynb), [Keyword expansion](https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/analytics_keyword_expansion_ads.ipynb)

You can check a full list of examples [here](https://github.com/superlinked/superlinked/tree/main/notebook).


## Run in production

With a single command you can run Superlinked as a REST API Server locally or in your cloud with [Superlinked Server](https://pypi.org/project/superlinked-server). Get data ingestion and query APIs, embedding model inference and deep vector database integrations for free!

Unify your evaluation, ingestion and serving stacks with a single declarative python codebase. Superlinked enables this by letting you define your data schema, vector indexes and the compute DAG that links them all at once and then choose the right executor for the task - in-memory or server.

If you are interested in learning more about running at scale, [Book a demo](https://links.superlinked.com/sl-repo-readme-form) for early access to our managed cloud.

### Supported Vector Databases

Superlinked stores *your vectors* in *your vector database*, with deep integrations for:

- [Redis](https://docs.superlinked.com/run-in-production/index-1/redis)
- [MongoDB](https://docs.superlinked.com/run-in-production/index-1/mongodb)
- [Qdrant](https://docs.superlinked.com/run-in-production/index-1/qdrant)
- [TopK](https://docs.superlinked.com/run-in-production/index-1/topk)
- [Which one should we support next?](https://github.com/superlinked/superlinked/discussions/41)

Curious about vector database pros & cons in general? Our community [compared 44 Vector Databases on 30+ features](https://superlinked.com/vector-db-comparison/).

## Logging

The Superlinked framework logs include contextual information, such as the process ID and package scope. Personally Identifiable Information (PII) is filtered out by default but can be exposed with the `SUPERLINKED_EXPOSE_PII` environment variable to `true`.

## Resources

- [Vector DB Comparison](https://superlinked.com/vector-db-comparison/): Open-source collaborative comparison of vector databases by Superlinked.
- [VectorHub](https://superlinked.com/vectorhub/): VectorHub is a free and open-sourced learning hub for people interested in adding vector retrieval to their ML stack


## Support

Need help? We're here to support you:
- Report a bug by creating an [issue](https://github.com/superlinked/superlinked/issues/new?assignees=kembala&labels=bug&projects=&template=bug_report.md&title=)
- Request a new feature [here](https://github.com/superlinked/superlinked/issues/new?assignees=kembala&labels=enhancement&projects=&template=feature_request.md&title=)
- Start a [discussion](https://github.com/superlinked/superlinked/discussions/new/choose) about your ideas or questions

Please create separate issues/discussions for each topic to help us better address your feedback. Thank you for contributing!
