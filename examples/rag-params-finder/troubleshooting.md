# Troubleshooting (SIE-focused)

Short FAQ for gallery readers. Full tables and recovery steps:
[project troubleshooting](https://github.com/neomatrix369/rag-params-finder/blob/main/docs/user-guide/troubleshooting.md)
(especially the SIE section) and
[sie-setup.md](https://github.com/neomatrix369/rag-params-finder/blob/main/docs/user-guide/sie-setup.md).

## `"sie": "disabled"` on `/health`

`SIE_ENABLED` is false or unset (default). Set `SIE_ENABLED=true`, set
`SIE_ENDPOINT`, reload the server.

## `"sie": "unreachable"` (or preflight fails)

- Wrong `SIE_ENDPOINT` (typo, http vs https, missing port)
- Gateway auth: set `SIE_API_KEY` and use `Authorization: Bearer …` on `/healthz`
- Local Docker: SIE not running, still warming up, or server-in-Docker needs
  `http://host.docker.internal:8720`
- Encode still returning **503** during model load — wait until encode returns **200**

## Sweep with `provider: sie` fails immediately

SIE guard runs preflight. Fix health/`SIE_ENABLED` first, then re-run. Index
requirements come from the **selected config**, not from `provider: sie` alone.
For [`configs/mongodb/example-sie.yaml`](https://github.com/neomatrix369/rag-params-finder/blob/main/configs/mongodb/example-sie.yaml)
(dense BGE-M3 / Stella-v5), create both indexes on the `chunks` collection:
`vector_index_1024` for dense 1024-dim embeddings, and `text_search_index` for
the separately swept sparse/hybrid retrievers — see project MongoDB setup.
Sparse-only models can need different indexes; do not treat `vector_index_1024`
as universal.

## `./start-services.sh` did not bring up SIE

Expected. Start a remote gateway or follow self-hosted Docker in
[sie-setup.md](https://github.com/neomatrix369/rag-params-finder/blob/main/docs/user-guide/sie-setup.md).

## Dashboard up but no SIE experiments

Use a SIE config such as
[`configs/mongodb/example-sie.yaml`](https://github.com/neomatrix369/rag-params-finder/blob/main/configs/mongodb/example-sie.yaml),
not a Voyage-only or local-only example.

## Still stuck?

1. [sie-setup.md](https://github.com/neomatrix369/rag-params-finder/blob/main/docs/user-guide/sie-setup.md) — known issues, warm-up, Aim UI  
2. [troubleshooting.md](https://github.com/neomatrix369/rag-params-finder/blob/main/docs/user-guide/troubleshooting.md) — indexes, Docker, storage  
3. Open an issue on
   [neomatrix369/rag-params-finder](https://github.com/neomatrix369/rag-params-finder/issues)
