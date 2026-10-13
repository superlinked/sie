"""Shared prefixes for the causal rerankers.

Every row a causal reranker scores is ``prefix + tail``: the chat template's
opening, the instruction and the query, then the document and the template's
closing tokens. In a rerank request the prefix is the same for every candidate;
when the reranker answers a yes-or-no question about a message (the question
as the query), it is the same on every request. Attention is causal, so a
prefix token's state depends on nothing after it: it is the same in every row
that starts with that prefix.

A shared forward therefore runs each distinct prefix once and each row's tail
once, as separate sequences of one packed stream, and gives every tail the
keys and values of its prefix to attend to. ``flash_attn_varlen_func`` takes
keys and values with their own ``cu_seqlens_k``, and with ``causal=True`` it
aligns the mask to the bottom right, so a tail of ``t`` tokens over ``p + t``
keys sees the whole prefix and its own tokens up to itself: the attention the
unshared row computes. Per layer the only extra work is one gather of the
packed keys and values into the order the sequences attend in
(``SharedForward.k_index``).

Across requests, the keys and values of a prefix are kept per layer in a
bounded cache (``PrefixCache``) the second time the prefix is seen, so a
question asked of one message at a time is encoded once. One cached token
holds ``2 * layers * kv_heads * head_dim`` values: 147 KB for
Qwen3-Reranker-4B in bfloat16 and 115 KB for the 0.6B.

Scores move by floating-point rounding against the unshared forward, as they
do when requests are batched together: a row's matrix products see a
different number of rows.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

# First sights the cache remembers, so that it can tell a second sight.
_SIGHTS = 1024


@dataclass
class PrefixEntry:
    """The keys and values of one prefix, per layer, as a tail attends to them."""

    # [layers, tokens, kv_heads, head_dim]; the keys after k-norm and the rotary embedding.
    keys: torch.Tensor
    values: torch.Tensor

    @property
    def tokens(self) -> int:
        return self.keys.shape[1]

    @property
    def nbytes(self) -> int:
        return self.keys.nbytes + self.values.nbytes


class PrefixCache:
    """Prefixes' keys and values, least recently used first out, within a byte budget.

    A prefix is worth keeping when it comes back: ``seen`` records a sight and
    says whether there was one before, and the caller stores the prefix on that
    second sight. A rerank request's prefix (its own query) is therefore never
    copied into the cache unless the same query is reranked again.

    Args:
        max_bytes: Device memory the cached keys and values may hold.
        max_entries: Prefixes the cache may hold.
    """

    def __init__(self, max_bytes: int, max_entries: int = 64) -> None:
        self._max_bytes = max_bytes
        self._max_entries = max_entries
        self._entries: OrderedDict[bytes, PrefixEntry] = OrderedDict()
        self._sights: OrderedDict[bytes, None] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.stores = 0
        self.evictions = 0

    def get(self, key: bytes) -> PrefixEntry | None:
        """The cached prefix, now the most recently used; None when it is not cached."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                self.hits += 1
            return entry

    def seen(self, key: bytes) -> bool:
        """Record a sight of ``key``; whether it had been seen before."""
        with self._lock:
            before = key in self._sights
            self._sights[key] = None
            self._sights.move_to_end(key)
            while len(self._sights) > _SIGHTS:
                self._sights.popitem(last=False)
            return before

    def fits(self, nbytes: int) -> bool:
        """Whether one prefix of this size may be cached at all."""
        return 0 < nbytes <= self._max_bytes and self._max_entries > 0

    def put(self, key: bytes, entry: PrefixEntry) -> None:
        """Cache a prefix, dropping the least recently used ones that no longer fit."""
        if not self.fits(entry.nbytes):
            return
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._entries[key] = entry
            self._bytes += entry.nbytes
            self.stores += 1
            while self._bytes > self._max_bytes or len(self._entries) > self._max_entries:
                _, dropped = self._entries.popitem(last=False)
                self._bytes -= dropped.nbytes
                self.evictions += 1

    def clear(self) -> None:
        """Drop every cached prefix and its memory."""
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def entry_ids(self) -> set[int]:
        """``id()`` of every cached prefix, for holders of per-prefix state to drop what the cache dropped."""
        with self._lock:
            return {id(entry) for entry in self._entries.values()}

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def held_bytes(self) -> int:
        return self._bytes


@dataclass
class SharedForward:
    """One packed forward in which rows attend to one copy of the prefix they share.

    The packed stream holds, in order, the prefixes this forward computes and
    then every row's own tokens: its tail when its prefix is computed here or
    cached, the whole row otherwise. Each of those is one query sequence. The
    keys a layer attends over are the cached prefixes' keys followed by the
    packed keys; ``k_index`` lists, sequence after sequence, the keys each one
    attends to (its prefix's, then its own).
    """

    input_ids: np.ndarray  # int64 [tokens]
    positions: np.ndarray  # int64 [tokens], each token's position in its row
    cu_seqlens_q: np.ndarray  # int32 [sequences + 1]
    cu_seqlens_k: np.ndarray  # int32 [sequences + 1]
    max_seqlen_q: int
    max_seqlen_k: int
    # int64 [cu_seqlens_k[-1]]; None when every sequence attends to the keys in place.
    k_index: np.ndarray | None
    last_index: np.ndarray  # int64 [rows], the packed position of each row's last token
    # Cached prefixes this forward attends to, in the order their keys precede the packed keys.
    cached: list[PrefixEntry] = field(default_factory=list)
    # Prefixes computed here that the cache should keep: (cache key, first packed token, one past the last).
    store: list[tuple[bytes, int, int]] = field(default_factory=list)


def prefix_key(prefix_ids: Sequence[int]) -> bytes:
    """The cache key of a prefix: its token ids."""
    return np.asarray(prefix_ids, dtype=np.int32).tobytes()


def plan_shared_forward(
    rows: Sequence[tuple[Sequence[int], Sequence[int]]], cache: PrefixCache | None, entry_bytes_per_token: int = 0
) -> SharedForward | None:
    """Lay out a forward that shares prefixes; None when no row can share one.

    A prefix is taken from the cache when it is there, computed once when at
    least two rows start with it, and computed once and kept when it is seen
    for the second time. A row whose prefix is none of these keeps it and runs
    as it does unshared.

    Args:
        rows: ``(prefix ids, tail ids)`` per row; an empty prefix never shares.
        cache: The cross-request cache; None shares inside the forward only.
        entry_bytes_per_token: Device bytes one cached prefix token takes, to
            skip keeping a prefix the cache could not hold.
    """
    # Rows of one query hold the same prefix list: its key is built once.
    keys_by_list: dict[int, bytes] = {}
    keys: list[bytes | None] = []
    for prefix, _ in rows:
        key = keys_by_list.get(id(prefix)) if len(prefix) else None
        if key is None and len(prefix):
            key = keys_by_list[id(prefix)] = prefix_key(prefix)
        keys.append(key)
    counts: dict[bytes, int] = {}
    for key in keys:
        if key is not None:
            counts[key] = counts.get(key, 0) + 1

    cached: dict[bytes, tuple[int, int]] = {}  # key -> its range in the cached keys
    cached_entries: list[PrefixEntry] = []
    computed: dict[bytes, Sequence[int]] = {}  # key -> prefix ids, in first-row order
    keep: set[bytes] = set()
    cached_tokens = 0
    for (prefix, _), key in zip(rows, keys, strict=True):
        if key is None or key in cached or key in computed:
            continue
        entry = cache.get(key) if cache is not None else None
        if entry is not None:
            cached[key] = (cached_tokens, cached_tokens + entry.tokens)
            cached_entries.append(entry)
            cached_tokens += entry.tokens
            continue
        again = cache is not None and cache.seen(key) and cache.fits(len(prefix) * entry_bytes_per_token)
        if again:
            keep.add(key)
        if again or counts[key] > 1:
            computed[key] = prefix
    if not cached and not computed:
        return None

    ids: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    q_lengths: list[int] = []
    k_lengths: list[int] = []
    k_parts: list[np.ndarray] = []
    packed: dict[bytes, tuple[int, int]] = {}  # key -> its range in the packed stream
    store: list[tuple[bytes, int, int]] = []
    total = 0
    for key, prefix in computed.items():
        n = len(prefix)
        ids.append(np.asarray(prefix, dtype=np.int64))
        positions.append(np.arange(n, dtype=np.int64))
        packed[key] = (total, total + n)
        if key in keep:
            store.append((key, total, total + n))
        q_lengths.append(n)
        k_lengths.append(n)
        k_parts.append(np.arange(cached_tokens + total, cached_tokens + total + n, dtype=np.int64))
        total += n
    last_index = np.empty(len(rows), dtype=np.int64)
    for i, ((prefix, tail), key) in enumerate(zip(rows, keys, strict=True)):
        if key in cached:
            start, end = cached[key]
        elif key in packed:
            start, end = packed[key]
            start, end = cached_tokens + start, cached_tokens + end
        else:
            start = end = 0
        shared = end - start
        own = np.asarray(tail if shared else [*prefix, *tail], dtype=np.int64)
        n = len(own)
        ids.append(own)
        positions.append(np.arange(shared, shared + n, dtype=np.int64))
        q_lengths.append(n)
        k_lengths.append(shared + n)
        if shared:
            k_parts.append(np.arange(start, end, dtype=np.int64))
        k_parts.append(np.arange(cached_tokens + total, cached_tokens + total + n, dtype=np.int64))
        total += n
        last_index[i] = total - 1

    k_index = np.concatenate(k_parts)
    in_place = len(k_index) == cached_tokens + total and bool((k_index == np.arange(len(k_index))).all())
    return SharedForward(
        input_ids=np.concatenate(ids),
        positions=np.concatenate(positions),
        cu_seqlens_q=np.concatenate(([0], np.cumsum(q_lengths))).astype(np.int32),
        cu_seqlens_k=np.concatenate(([0], np.cumsum(k_lengths))).astype(np.int32),
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        k_index=None if in_place else k_index,
        last_index=last_index,
        cached=cached_entries,
        store=store,
    )
