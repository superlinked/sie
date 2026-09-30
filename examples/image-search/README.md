# Find the product photo a shopper describes

The runnable example behind [superlinked.com/image-search](https://superlinked.com/image-search).
That page's sources are in its [SOURCES.md](https://superlinked.com/reference/image-search/SOURCES.md).

## What this shows

A shopper types "brown leather sofa". The catalogue record says "Sofa, SKU
4471"; the colour and the material are in the photo and nowhere else. This
example holds a recorded run in which SIE's SigLIP so400m and three hosted
alternatives search a real product catalogue by those words.

- **The catalogue.** 2,573 main product photos from
  [Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
  (CC BY 4.0), each with one colour, one material and one product type. The
  labels come from ABO's own listing fields, folded into the words a shopper
  types, and were kept only where Claude Sonnet 5, shown the photo without its
  label, named the same three.
- **309 questions.** Each is a label written as a shopper types it, such as
  "grey leather sofa", and only for labels where the catalogue also holds a near
  miss (the same sofa in black, a grey fabric sofa). A photo is a right answer
  when its colour, material and type all match.
- **Flickr30k and MS-COCO**, the public text-to-image test sets, scored per
  caption.

Each product ranks every photo by cosine similarity. The figure is the share of
questions whose first result is a right product.

| Product | Right product first (of 309) |
| --- | --- |
| SIE SigLIP so400m (`google/siglip-so400m-patch14-224`) | 84.1% |
| Cohere Embed v4 | 84.1% |
| Voyage multimodal-3.5 | 75.1% |
| GPT-6 Luna captions + OpenAI text-embedding-3-small | 53.4% |

SIE is ahead of Voyage (47 questions only SIE got right, 19 only Voyage did;
p < 0.001) and level with Cohere (23 and 23). On Flickr30k and MS-COCO, which are
street and scene photos rather than products, Cohere and Voyage rank the right
image first more often than SIE; `score.py` prints those too.

## Models and settings

| Product | Called as | Photo | Query |
| --- | --- | --- | --- |
| SIE SigLIP so400m | `google/siglip-so400m-patch14-224` on the open-source SIE server | JPEG bytes | the question |
| Voyage multimodal-3.5 | Voyage `multimodalembeddings` | `input_type="document"` | `input_type="query"` |
| Cohere Embed v4 | Cohere v2 embed, `embed-v4.0`, float | `search_document` | `search_query` |
| GPT-6 Luna captions | GPT-6 Luna captions each photo once (`detail="low"`), then `text-embedding-3-small` embeds the caption | the caption | `text-embedding-3-small` |

Photos went to every product as the same 1,024-pixel JPEGs. All ran on 30
September 2026.

## Run it

Download the recorded run, then score it. Neither step needs a key:

```sh
uv sync
uv run python fetch.py
uv run python score.py
```

`fetch.py` pulls the evidence from the public HuggingFace dataset
[superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence),
pinned to one revision, and checks every file against the id the dataset lists.
It holds the frozen catalogue and questions, SIE's recorded vectors, every
product's recorded top 20 for every question, and every product's rank of the
right image on Flickr30k and MS-COCO.

`score.py` ranks all 2,573 photos for every question from SIE's vectors, fails
unless that matches the recording, then counts each product's right first
results, compares each with SIE by exact McNemar test, and prints the public-set
figures. It fails if a figure differs from the page.

To embed the catalogue yourself, fetch the photos too, then point `run.py` at
SIE Cloud or at your own server:

```sh
uv run python fetch.py --photos
sie-server serve -m google/siglip-so400m-patch14-224      # or use SIE Cloud with SIE_API_KEY
uv run python run.py --base-url http://localhost:8080
uv run python score.py --vectors run-output/vectors
```

## Credit

The photos and listing fields are from Amazon Berkeley Objects, by Amazon.com,
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset's
`ATTRIBUTION.md` lists the changes made to them.

## What this does not establish

- **One catalogue.** ABO is furniture, shoes, bags and home goods on plain
  backgrounds. A catalogue of lifestyle photos, or one where colour is rarely
  asked for, may rank differently.
- **The labels are ABO's, checked by one model.** A photo stayed only where
  Claude Sonnet 5 agreed with its label, and 15% of the kept photos were read by
  eye; a label can still be arguable at the edges (a clock face in two colours).
- **Public images may be in any model's training data.** ABO has been public
  since 2021.
