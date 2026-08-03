# Sources and rights

## Supermarket Shelves Dataset

- Publisher: Humans in the Loop
- Kaggle dataset: [`humansintheloop/supermarket-shelves-dataset`](https://www.kaggle.com/datasets/humansintheloop/supermarket-shelves-dataset)
- Dataset version: 2, updated June 6, 2023
- Publisher catalog: [Humans in the Loop datasets](https://humansintheloop.org/resources/datasets/)
- License: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)
- Archive SHA-256: `988c30cc94c38660098114d7b8691e5a3f34123b8923e3772b64d6f5c7b67dd1`
- Kaggle metadata SHA-256: `fdfd21f8018b3720690b957fe332dc80638f1bb06da65cda19431849a83d48d0`

The publisher describes 45 images with 11,743 `Product` and `Price` boxes. The Kaggle data page marks the dataset copyright-free and dedicated under CC0. CC0 permits copying, modification, and commercial use without asking permission.

The CC0 deed does not grant trademark, publicity, or privacy rights. Product and retailer marks remain with their owners. This example uses the image as a technical fixture and does not imply endorsement.

## Included source case

- Image: original dataset file `images/042.jpg`, renamed `assets/042-pharmacy-oos-sign.jpg`
- Image SHA-256: `a8654b3a7c2be143fb788f91aeef0c09b23feaee0984151eefed95f6c7a96f63`
- Dimensions: 4032 by 3024 pixels
- Annotation: original `annotations/042.jpg.json`, renamed `assets/042-pharmacy-oos-sign.jpg.json`
- Annotation SHA-256: `5ff910b8b63f510f6e8a8ca807502c52fb0c2487eaf12799c9c8e65439d86c81`
- Annotation counts: 37 `Product` boxes and 34 `Price` boxes

The two recorded OCR crops are deterministic derivatives of DINO candidate boxes on this image. Their source boxes, coordinates, display scales, and checksums are recorded in `lighton-ocr-derived-results.json` and `source-manifest.json`.

## Model records

The model repositories and launch revisions are listed in the README and `retail_shelf_audit/config.py`. The recorded outputs are generated facts rather than dataset source material. Their checksums appear in `source-manifest.json`.
