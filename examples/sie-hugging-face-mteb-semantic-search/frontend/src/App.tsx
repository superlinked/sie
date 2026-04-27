import React, { useEffect, useState } from "react";
import Markdown from "react-markdown";

type DownloadResponse = {
  storage_id: string;
  stored_count: number;
};

type ModelSummary = {
  hf_id: string;
  author?: string | null;
  created_at: string;
  downloads_30d?: number | null;
  likes?: number | null;
  pipeline_tag?: string | null;
  short_description?: string | null;
};

type ModelCard = {
  hf_id: string;
  author?: string | null;
  sha?: string | null;
  created_at_hf?: string;
  last_modified?: string;
  created_at: string;

  private?: boolean | null;
  disabled?: boolean | null;

  downloads?: number | null;
  downloads_all_time?: number | null;
  downloads_30d?: number | null;
  likes?: number | null;
  trending_score?: number | null;

  tags?: string[] | null;
  pipeline_tag?: string | null;
  library_name?: string | null;
  mask_token?: string | null;

  config?: unknown;
  card_data?: unknown;

  mteb_scores?: {
    task_name: string;
    main_score: number | null;
  }[] | null;

  short_description?: string | null;
  long_description?: string | null;
};

type SearchResult = {
  hf_id: string;
  distance: number;
  short_description?: string | null;
};

type RerankSearchResult = {
  hf_id: string;
  rerank_distance: number | null;
  short_distance: number;
  short_description?: string | null;
};

type TabId = "download" | "browse" | "search" | "search-rerank";

const API_BASE = "http://localhost:8000/api";
const DEMO_STORAGE_ID = "demo";

/**
 * Fetch a model's README live from the backend (which proxies HuggingFace).
 * README is no longer stored in SQLite — see overview.md.
 */
const useLiveReadme = (hfId: string | null) => {
  const [readme, setReadme] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!hfId) {
      setReadme(null);
      setError(null);
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    setReadme(null);
    (async () => {
      try {
        const res = await fetch(
          `${API_BASE}/models/readme/${encodeURI(hfId)}`
        );
        if (!res.ok) {
          if (!cancelled) setError("README not available on HuggingFace");
          return;
        }
        const data: { readme?: string } = await res.json();
        if (!cancelled) setReadme(data.readme ?? "");
      } catch {
        if (!cancelled) setError("Failed to load README from HuggingFace");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [hfId]);

  return { readme, loading, error };
};

const ReadmeBlock: React.FC<{ hfId: string }> = ({ hfId }) => {
  const { readme, loading, error } = useLiveReadme(hfId);
  return (
    <div className="detail-block">
      <div className="detail-label">README</div>
      {loading && (
        <p className="placeholder">Loading README from HuggingFace…</p>
      )}
      {error && !loading && <p className="placeholder">{error}</p>}
      {readme && !loading && (
        <div className="readme-content">
          <Markdown>{readme}</Markdown>
        </div>
      )}
    </div>
  );
};

const DownloadView: React.FC = () => {
  const [storageId, setStorageId] = useState(DEMO_STORAGE_ID);
  const [limit, setLimit] = useState(30);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DownloadResponse | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/models/download`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ storage_id: storageId, limit })
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Request failed with status ${res.status}`);
      }

      const data: DownloadResponse = await res.json();
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unknown error while downloading models"
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <h1>SIE – Download LLM Cards</h1>
      <p className="subtitle">
        Select the top MTEB-benchmarked embedding models (ranked by number of benchmark
        tasks), then fetch their Hugging Face metadata and persist under a storage id.
      </p>

      <form onSubmit={handleSubmit} className="form">
        <label className="field">
          <span>Storage ID</span>
          <input
            type="text"
            value={storageId}
            onChange={(e) => setStorageId(e.target.value)}
            placeholder="e.g. test01"
            required
          />
        </label>

        <label className="field">
          <span>How many models to download (max 200)</span>
          <input
            type="number"
            min={1}
            max={200}
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
          />
        </label>

        <button type="submit" disabled={loading}>
          {loading ? "Downloading..." : "Download models"}
        </button>
      </form>

      {error && <div className="alert alert-error">{error}</div>}
      {result && (
        <div className="alert alert-success">
          <div>
            <strong>Done.</strong> Stored {result.stored_count} models under storage id{" "}
            <code>{result.storage_id}</code>.
          </div>
        </div>
      )}
    </>
  );
};

type GenerateModalProps = {
  storageId: string;
  hfId: string;
  onClose: () => void;
  onDescriptionsSaved: (short: string, long: string) => void;
};

const GenerateModal: React.FC<GenerateModalProps> = ({
  storageId,
  hfId,
  onClose,
  onDescriptionsSaved,
}) => {
  const [promptText, setPromptText] = useState("");
  const [promptLoading, setPromptLoading] = useState(true);
  const [promptError, setPromptError] = useState<string | null>(null);

  const [modelName, setModelName] = useState("google/gemini-3.1-pro-preview");

  const [detailed, setDetailed] = useState("");
  const [long, setLong] = useState("");
  const [short, setShort] = useState("");

  const [detailedLoading, setDetailedLoading] = useState(false);
  const [longLoading, setLongLoading] = useState(false);
  const [shortLoading, setShortLoading] = useState(false);

  const [saveStatus, setSaveStatus] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setPromptLoading(true);
      setPromptError(null);
      try {
        const res = await fetch(`${API_BASE}/generate/render-prompt`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ storage_id: storageId, hf_id: hfId }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        if (!cancelled) setPromptText(data.prompt_text);
      } catch (err) {
        if (!cancelled)
          setPromptError(
            err instanceof Error ? err.message : "Failed to load prompt"
          );
      } finally {
        if (!cancelled) setPromptLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [storageId, hfId]);

  const handleGenerateDetailed = async () => {
    setDetailedLoading(true);
    try {
      const res = await fetch(`${API_BASE}/generate/detailed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt_text: promptText, model: modelName || undefined }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setDetailed(data.text);
    } catch (err) {
      setDetailed(
        `Error: ${err instanceof Error ? err.message : "Generation failed"}`
      );
    } finally {
      setDetailedLoading(false);
    }
  };

  const handleGenerateLong = async () => {
    if (!detailed) return;
    setLongLoading(true);
    try {
      const res = await fetch(`${API_BASE}/generate/long`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          storage_id: storageId,
          hf_id: hfId,
          detailed_description: detailed,
          model: modelName || undefined,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setLong(data.text);
    } catch (err) {
      setLong(
        `Error: ${err instanceof Error ? err.message : "Generation failed"}`
      );
    } finally {
      setLongLoading(false);
    }
  };

  const handleGenerateShort = async () => {
    if (!detailed) return;
    setShortLoading(true);
    try {
      const res = await fetch(`${API_BASE}/generate/short`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          storage_id: storageId,
          hf_id: hfId,
          detailed_description: detailed,
          model: modelName || undefined,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setShort(data.text);
    } catch (err) {
      setShort(
        `Error: ${err instanceof Error ? err.message : "Generation failed"}`
      );
    } finally {
      setShortLoading(false);
    }
  };

  const handleSave = async () => {
    if (!short && !long) return;
    setSaveStatus("Saving...");
    try {
      const res = await fetch(`${API_BASE}/generate/save`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          storage_id: storageId,
          hf_id: hfId,
          short_description: short || undefined,
          long_description: long || undefined,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      setSaveStatus("Saved");
      if (short || long) onDescriptionsSaved(short, long);
    } catch {
      setSaveStatus("Save failed");
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Generate Descriptions</h2>
          <button type="button" className="modal-close" onClick={onClose}>
            &times;
          </button>
        </div>

        <div className="modal-body">
          <div className="modal-field">
            <span className="detail-label">Model</span>
            <span className="code-like">{hfId}</span>
          </div>

          <div className="modal-field">
            <label className="detail-label" htmlFor="or-model">
              OpenRouter AI model
            </label>
            <input
              id="or-model"
              type="text"
              className="input-model-name"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
            />
          </div>

          <div className="modal-section">
            <label className="detail-label">
              Detailed description prompt (editable)
            </label>
            {promptLoading ? (
              <p className="empty-state">Loading prompt...</p>
            ) : promptError ? (
              <div className="alert alert-error">{promptError}</div>
            ) : (
              <textarea
                className="modal-textarea-large"
                value={promptText}
                onChange={(e) => setPromptText(e.target.value)}
              />
            )}
          </div>

          <div className="modal-buttons">
            <button
              type="button"
              className="btn-generate"
              disabled={detailedLoading || !promptText}
              onClick={handleGenerateDetailed}
            >
              {detailedLoading ? "Generating..." : "Generate 6K description"}
            </button>
            <button
              type="button"
              className="btn-generate"
              disabled={longLoading || !detailed}
              onClick={handleGenerateLong}
            >
              {longLoading ? "Generating..." : "Generate 2K description"}
            </button>
            <button
              type="button"
              className="btn-generate"
              disabled={shortLoading || !detailed}
              onClick={handleGenerateShort}
            >
              {shortLoading ? "Generating..." : "Generate 200 char description"}
            </button>
          </div>

          <div className="modal-section">
            <label className="detail-label">
              6K detailed description
              {detailed && (
                <span className="char-count">{detailed.length} chars</span>
              )}
            </label>
            <textarea
              className="modal-textarea-medium"
              value={detailed}
              onChange={(e) => setDetailed(e.target.value)}
              placeholder="Click 'Generate 6K description' above..."
            />
          </div>

          <div className="modal-section">
            <label className="detail-label">
              2K long description
              {long && (
                <span className="char-count">{long.length} chars</span>
              )}
            </label>
            <textarea
              className="modal-textarea-small"
              value={long}
              onChange={(e) => setLong(e.target.value)}
              placeholder="Click 'Generate 2K description' above..."
            />
          </div>

          <div className="modal-section">
            <label className="detail-label">
              200 char short description
              {short && (
                <span className="char-count">{short.length} chars</span>
              )}
            </label>
            <textarea
              className="modal-textarea-small"
              value={short}
              onChange={(e) => setShort(e.target.value)}
              placeholder="Click 'Generate 200 char description' above..."
            />
          </div>

          <div className="modal-footer">
            <button
              type="button"
              disabled={!short && !long}
              onClick={handleSave}
            >
              Save descriptions
            </button>
            {saveStatus && (
              <span
                className={
                  saveStatus === "Saved" ? "save-ok" : "save-info"
                }
              >
                {saveStatus}
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const SearchView: React.FC = () => {
  const [storageId, setStorageId] = useState("test01");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [selectedDetail, setSelectedDetail] = useState<ModelCard | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setResults([]);
    setSelectedDetail(null);

    try {
      const res = await fetch(`${API_BASE}/search/semantic`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          storage_id: storageId,
          query: query.trim(),
          n_results: 20,
        }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Request failed with status ${res.status}`);
      }
      const data: { results: SearchResult[] } = await res.json();
      setResults(data.results || []);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unknown error during search"
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSelectResult = async (hfId: string) => {
    setDetailLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        storage_id: storageId,
        hf_id: hfId,
      });
      const res = await fetch(`${API_BASE}/models/detail?${params.toString()}`);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Request failed with status ${res.status}`);
      }
      const data: ModelCard = await res.json();
      setSelectedDetail(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load model details"
      );
    } finally {
      setDetailLoading(false);
    }
  };

  if (selectedDetail) {
    return (
      <>
        <h1>Search Result Details</h1>
        <p className="subtitle">
          Full details for <code>{selectedDetail.hf_id}</code> in storage{" "}
          <code>{storageId}</code>.
        </p>

        <button type="button" onClick={() => setSelectedDetail(null)}>
          Back to search results
        </button>

        <div className="detail">
          <div className="detail-row">
            <span className="detail-label">HF ID</span>
            <span className="detail-value code-like">{selectedDetail.hf_id}</span>
          </div>
          {selectedDetail.author && (
            <div className="detail-row">
              <span className="detail-label">Author</span>
              <span className="detail-value">{selectedDetail.author}</span>
            </div>
          )}
          <div className="detail-row">
            <span className="detail-label">Downloads (30d)</span>
            <span className="detail-value">
              {selectedDetail.downloads_30d != null
                ? selectedDetail.downloads_30d.toLocaleString()
                : "—"}
            </span>
          </div>
          <div className="detail-row">
            <span className="detail-label">Likes</span>
            <span className="detail-value">
              {selectedDetail.likes != null
                ? selectedDetail.likes.toLocaleString()
                : "—"}
            </span>
          </div>
          {(selectedDetail.pipeline_tag || selectedDetail.library_name) && (
            <div className="detail-row">
              <span className="detail-label">Pipeline / Library</span>
              <span className="detail-value">
                {selectedDetail.pipeline_tag ?? "—"}
                {selectedDetail.pipeline_tag && selectedDetail.library_name
                  ? " • "
                  : ""}
                {selectedDetail.library_name ?? ""}
              </span>
            </div>
          )}

          <div className="detail-block">
            <div className="detail-label">Short description</div>
            <p className="detail-text">
              {selectedDetail.short_description || (
                <span className="placeholder">Not generated yet</span>
              )}
            </p>
          </div>

          <div className="detail-block">
            <div className="detail-label">Long description</div>
            <p className="detail-text">
              {selectedDetail.long_description || (
                <span className="placeholder">Not generated yet</span>
              )}
            </p>
          </div>

          <ReadmeBlock hfId={selectedDetail.hf_id} />
        </div>

        <button type="button" onClick={() => setSelectedDetail(null)}>
          Back to search results
        </button>
      </>
    );
  }

  return (
    <>
      <h1>Search</h1>
      <p className="subtitle">
        Describe what the model should do and find the best matching embedding
        models by semantic similarity. Use storage <code>{DEMO_STORAGE_ID}</code> to
        try the bundled local demo without SIE or OpenRouter credentials.
      </p>

      <form onSubmit={handleSearch} className="form">
        <label className="field">
          <span>Storage ID</span>
          <input
            type="text"
            value={storageId}
            onChange={(e) => setStorageId(e.target.value)}
            placeholder={`e.g. ${DEMO_STORAGE_ID}`}
            required
          />
        </label>

        <label className="field">
          <span>What should the model do?</span>
          <textarea
            className="search-textarea"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="evaluate medical pictures"
            rows={3}
            required
          />
        </label>

        <button type="submit" disabled={loading || !query.trim()}>
          {loading ? "Searching..." : "Find models"}
        </button>
      </form>

      {error && <div className="alert alert-error">{error}</div>}

      {detailLoading && <p className="empty-state">Loading model details...</p>}

      {results.length > 0 && !detailLoading && (
        <div className="list">
          {results.map((r, idx) => (
            <button
              key={r.hf_id}
              type="button"
              className="list-item"
              onClick={() => handleSelectResult(r.hf_id)}
            >
              <div className="list-title">
                <span className="search-rank">#{idx + 1}</span> {r.hf_id}
                <span className="search-result-distance">
                  {((1 - r.distance) * 100).toFixed(1)}% match
                </span>
              </div>
              {r.short_description && (
                <div className="list-meta">
                  <span>{r.short_description}</span>
                </div>
              )}
            </button>
          ))}
        </div>
      )}

      {!loading && results.length === 0 && (
        <p className="empty-state">
          Enter a description and click "Find models" to search.
        </p>
      )}
    </>
  );
};

const SearchWithRerankingView: React.FC = () => {
  const [storageId, setStorageId] = useState(DEMO_STORAGE_ID);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<RerankSearchResult[]>([]);
  const [selectedDetail, setSelectedDetail] = useState<ModelCard | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setResults([]);
    setSelectedDetail(null);

    try {
      const res = await fetch(`${API_BASE}/search/semantic-rerank`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          storage_id: storageId,
          query: query.trim(),
          n_results: 20,
        }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Request failed with status ${res.status}`);
      }
      const data: { results: RerankSearchResult[] } = await res.json();
      setResults(data.results || []);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unknown error during search"
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSelectResult = async (hfId: string) => {
    setDetailLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        storage_id: storageId,
        hf_id: hfId,
      });
      const res = await fetch(`${API_BASE}/models/detail?${params.toString()}`);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Request failed with status ${res.status}`);
      }
      const data: ModelCard = await res.json();
      setSelectedDetail(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load model details"
      );
    } finally {
      setDetailLoading(false);
    }
  };

  if (selectedDetail) {
    return (
      <>
        <h1>Search Result Details</h1>
        <p className="subtitle">
          Full details for <code>{selectedDetail.hf_id}</code> in storage{" "}
          <code>{storageId}</code>.
        </p>

        <button type="button" onClick={() => setSelectedDetail(null)}>
          Back to search results
        </button>

        <div className="detail">
          <div className="detail-row">
            <span className="detail-label">HF ID</span>
            <span className="detail-value code-like">{selectedDetail.hf_id}</span>
          </div>
          {selectedDetail.author && (
            <div className="detail-row">
              <span className="detail-label">Author</span>
              <span className="detail-value">{selectedDetail.author}</span>
            </div>
          )}
          <div className="detail-row">
            <span className="detail-label">Downloads (30d)</span>
            <span className="detail-value">
              {selectedDetail.downloads_30d != null
                ? selectedDetail.downloads_30d.toLocaleString()
                : "—"}
            </span>
          </div>
          <div className="detail-row">
            <span className="detail-label">Likes</span>
            <span className="detail-value">
              {selectedDetail.likes != null
                ? selectedDetail.likes.toLocaleString()
                : "—"}
            </span>
          </div>

          <div className="detail-block">
            <div className="detail-label">Short description</div>
            <p className="detail-text">
              {selectedDetail.short_description || (
                <span className="placeholder">Not generated yet</span>
              )}
            </p>
          </div>

          <div className="detail-block">
            <div className="detail-label">Long description</div>
            <p className="detail-text">
              {selectedDetail.long_description || (
                <span className="placeholder">Not generated yet</span>
              )}
            </p>
          </div>

          <ReadmeBlock hfId={selectedDetail.hf_id} />
        </div>

        <button type="button" onClick={() => setSelectedDetail(null)}>
          Back to search results
        </button>
      </>
    );
  }

  return (
    <>
      <h1>Search with Reranking</h1>
      <p className="subtitle">
        First we find candidates by short-description similarity, then rerank
        them by long-description similarity for higher-quality results. Use storage{" "}
        <code>{DEMO_STORAGE_ID}</code> to try the bundled local demo without SIE or
        OpenRouter credentials.
      </p>

      <form onSubmit={handleSearch} className="form">
        <label className="field">
          <span>Storage ID</span>
          <input
            type="text"
            value={storageId}
            onChange={(e) => setStorageId(e.target.value)}
            placeholder={`e.g. ${DEMO_STORAGE_ID}`}
            required
          />
        </label>

        <label className="field">
          <span>What should the model do?</span>
          <textarea
            className="search-textarea"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="evaluate medical pictures"
            rows={3}
            required
          />
        </label>

        <button type="submit" disabled={loading || !query.trim()}>
          {loading ? "Searching..." : "Find models"}
        </button>
      </form>

      {error && <div className="alert alert-error">{error}</div>}

      {detailLoading && <p className="empty-state">Loading model details...</p>}

      {results.length > 0 && !detailLoading && (
        <div className="list">
          {results.map((r, idx) => {
            const primary =
              r.rerank_distance != null ? r.rerank_distance : r.short_distance;
            return (
              <button
                key={r.hf_id}
                type="button"
                className="list-item"
                onClick={() => handleSelectResult(r.hf_id)}
              >
                <div className="list-title">
                  <span className="search-rank">#{idx + 1}</span> {r.hf_id}
                  <span className="search-result-distance">
                    {((1 - primary) * 100).toFixed(1)}% match
                    {r.rerank_distance == null && " (short only)"}
                  </span>
                </div>
                {r.short_description && (
                  <div className="list-meta">
                    <span>{r.short_description}</span>
                  </div>
                )}
                <div className="list-meta">
                  <span>
                    short {((1 - r.short_distance) * 100).toFixed(1)}%
                    {r.rerank_distance != null && (
                      <>
                        {" "}
                        · long {((1 - r.rerank_distance) * 100).toFixed(1)}%
                      </>
                    )}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      )}

      {!loading && results.length === 0 && (
        <p className="empty-state">
          Enter a description and click "Find models" to search.
        </p>
      )}
    </>
  );
};

const BrowseView: React.FC = () => {
  const [storageId, setStorageId] = useState(DEMO_STORAGE_ID);
  const [hfIdFilter, setHfIdFilter] = useState("");
  const [maxCount, setMaxCount] = useState(30);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [selected, setSelected] = useState<ModelCard | null>(null);
  const [showGenerateModal, setShowGenerateModal] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSelected(null);

    try {
      const params = new URLSearchParams({ storage_id: storageId });
      if (hfIdFilter.trim()) {
        params.set("hf_id", hfIdFilter.trim());
      }
      params.set("max_count", String(maxCount));

      const res = await fetch(`${API_BASE}/models/search?${params.toString()}`);

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Request failed with status ${res.status}`);
      }

      const data: { models: ModelSummary[] } = await res.json();
      setModels(data.models || []);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unknown error while loading models"
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSelect = async (summary: ModelSummary) => {
    setDetailLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        storage_id: storageId,
        hf_id: summary.hf_id,
      });
      const res = await fetch(`${API_BASE}/models/detail?${params.toString()}`);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Request failed with status ${res.status}`);
      }
      const data: ModelCard = await res.json();
      setSelected(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load model details"
      );
    } finally {
      setDetailLoading(false);
    }
  };

  if (selected) {
    return (
      <>
        <h1>LLM Card Details</h1>
        <p className="subtitle">
          Full details for <code>{selected.hf_id}</code> in storage{" "}
          <code>{storageId}</code>.
        </p>

        <button type="button" onClick={() => setSelected(null)}>
          Back to results
        </button>

        <div className="detail">
          <div className="detail-row">
            <span className="detail-label">HF ID</span>
            <span className="detail-value code-like">{selected.hf_id}</span>
          </div>
          {selected.author && (
            <div className="detail-row">
              <span className="detail-label">Author</span>
              <span className="detail-value">{selected.author}</span>
            </div>
          )}
          <div className="detail-row">
            <span className="detail-label">Created at (HF)</span>
            <span className="detail-value">
              {selected.created_at_hf
                ? new Date(selected.created_at_hf).toLocaleString()
                : "—"}
            </span>
          </div>
          {selected.last_modified && (
            <div className="detail-row">
              <span className="detail-label">Last modified (HF)</span>
              <span className="detail-value">
                {new Date(selected.last_modified).toLocaleString()}
              </span>
            </div>
          )}
          <div className="detail-row">
            <span className="detail-label">Stored at</span>
            <span className="detail-value">
              {selected.created_at
                ? new Date(selected.created_at).toLocaleString()
                : "—"}
            </span>
          </div>
          <div className="detail-row">
            <span className="detail-label">Downloads (30d)</span>
            <span className="detail-value">
              {selected.downloads_30d != null
                ? selected.downloads_30d.toLocaleString()
                : "—"}
            </span>
          </div>
          <div className="detail-row">
            <span className="detail-label">Downloads (all time)</span>
            <span className="detail-value">
              {selected.downloads_all_time != null
                ? selected.downloads_all_time.toLocaleString()
                : "—"}
            </span>
          </div>
          <div className="detail-row">
            <span className="detail-label">Likes</span>
            <span className="detail-value">
              {selected.likes != null ? selected.likes.toLocaleString() : "—"}
            </span>
          </div>
          {(selected.pipeline_tag || selected.library_name) && (
            <div className="detail-row">
              <span className="detail-label">Pipeline / Library</span>
              <span className="detail-value">
                {selected.pipeline_tag ?? "—"}
                {selected.pipeline_tag && selected.library_name ? " • " : ""}
                {selected.library_name ?? ""}
              </span>
            </div>
          )}
          {selected.tags && selected.tags.length > 0 && (
            <div className="detail-row">
              <span className="detail-label">Tags</span>
              <span className="detail-value">
                {selected.tags.join(", ")}
              </span>
            </div>
          )}

          {selected.mteb_scores && selected.mteb_scores.length > 0 && (() => {
            const taskScores = [...selected.mteb_scores].sort(
              (a, b) => (b.main_score ?? 0) - (a.main_score ?? 0)
            );

            return (
              <div className="detail-block">
                <div className="detail-label">
                  MTEB Scores ({selected.mteb_scores!.length} tasks)
                </div>
                <div className="mteb-table-wrap">
                  <table className="mteb-table">
                    <thead>
                      <tr>
                        <th>Task</th>
                        <th>Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {taskScores.map((r) => (
                        <tr key={r.task_name}>
                          <td>{r.task_name}</td>
                          <td>
                            {r.main_score != null
                              ? (r.main_score * 100).toFixed(2) + "%"
                              : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })()}

          <div className="detail-block">
            <div className="detail-block-header">
              <div className="detail-label">Short description</div>
              <button
                type="button"
                className="btn-generate"
                onClick={() => setShowGenerateModal(true)}
              >
                Generate descriptions
              </button>
            </div>
            <p className="detail-text">
              {selected.short_description || <span className="placeholder">Not generated yet</span>}
            </p>
          </div>

          <div className="detail-block">
            <div className="detail-label">Long description</div>
            <p className="detail-text">
              {selected.long_description || <span className="placeholder">Not generated yet</span>}
            </p>
          </div>

          {showGenerateModal && (
            <GenerateModal
              storageId={storageId}
              hfId={selected.hf_id}
              onClose={() => setShowGenerateModal(false)}
              onDescriptionsSaved={(s, l) => {
                setSelected({
                  ...selected,
                  short_description: s || selected.short_description,
                  long_description: l || selected.long_description,
                });
              }}
            />
          )}

          <ReadmeBlock hfId={selected.hf_id} />
        </div>

        <button type="button" onClick={() => setSelected(null)}>
          Back to results
        </button>
      </>
    );
  }

  return (
    <>
      <h1>Browse LLM Cards</h1>
      <p className="subtitle">
        Enter a storage ID and optional LLM ID to list stored cards and inspect their
        details. The bundled <code>{DEMO_STORAGE_ID}</code> storage works locally after
        seeding the demo data.
      </p>

      <form onSubmit={handleSearch} className="form">
        <label className="field">
          <span>Storage ID</span>
          <input
            type="text"
            value={storageId}
            onChange={(e) => setStorageId(e.target.value)}
            placeholder={`e.g. ${DEMO_STORAGE_ID}`}
            required
          />
        </label>

        <label className="field">
          <span>LLM ID (optional)</span>
          <input
            type="text"
            value={hfIdFilter}
            onChange={(e) => setHfIdFilter(e.target.value)}
            placeholder="Filter by HF model id"
          />
        </label>

        <label className="field">
          <span>Max item count</span>
          <input
            type="number"
            value={maxCount}
            onChange={(e) => setMaxCount(Math.max(1, Number(e.target.value)))}
            min={1}
            max={10000}
          />
        </label>

        <button type="submit" disabled={loading}>
          {loading ? "Searching..." : "Search cards"}
        </button>
      </form>

      {error && <div className="alert alert-error">{error}</div>}

      {detailLoading && <p className="empty-state">Loading model details…</p>}

      {models.length > 0 && !detailLoading && (
        <div className="list">
          {models.map((m) => (
            <button
              key={`${m.hf_id}-${m.created_at}`}
              type="button"
              className="list-item"
              onClick={() => handleSelect(m)}
            >
              <div className="list-title">{m.hf_id}</div>
              <div className="list-meta">
                <span>
                  30d downloads:{" "}
                  {m.downloads_30d != null ? m.downloads_30d.toLocaleString() : "—"}
                </span>
                <span>
                  Stored:{" "}
                  {m.created_at
                    ? new Date(m.created_at).toLocaleDateString()
                    : "—"}
                </span>
              </div>
            </button>
          ))}
        </div>
      )}

      {!loading && models.length === 0 && (
        <p className="empty-state">
          No cards loaded yet for this storage. Try running a download first.
        </p>
      )}
    </>
  );
};

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabId>("search-rerank");

  return (
    <div className="page">
      <div className="card">
        <div className="tabs">
          <button
            type="button"
            className={activeTab === "search-rerank" ? "tab active" : "tab"}
            onClick={() => setActiveTab("search-rerank")}
          >
            Search with Reranking
          </button>
          <button
            type="button"
            className={activeTab === "search" ? "tab active" : "tab"}
            onClick={() => setActiveTab("search")}
          >
            Simple search
          </button>
          <button
            type="button"
            className={activeTab === "download" ? "tab active" : "tab"}
            onClick={() => setActiveTab("download")}
          >
            Download LLM Cards
          </button>
          <button
            type="button"
            className={activeTab === "browse" ? "tab active" : "tab"}
            onClick={() => setActiveTab("browse")}
          >
            Browse LLM Cards
          </button>
        </div>

        {activeTab === "search-rerank" ? (
          <SearchWithRerankingView />
        ) : activeTab === "search" ? (
          <SearchView />
        ) : activeTab === "download" ? (
          <DownloadView />
        ) : (
          <BrowseView />
        )}
      </div>
    </div>
  );
};
