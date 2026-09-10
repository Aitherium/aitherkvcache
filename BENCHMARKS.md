# Benchmarks

Measured results for the models and kernels published from this repository. Every
number here came from the code in this repo or from a stage of `awembed`, run against
real data — not from a paper, and not from a vendor's summary table.

---

## `aither-code-embed-0.6b` vs general-purpose embedders, on real documents

*Measured 2026-09-10.*

The model in the [`aither-code-embed-v1`](../../releases/tag/aither-code-embed-v1)
release is a 0.6B student distilled for **code search** — trained on a large teacher's
margins over a code corpus, exported to Q8_0 GGUF (639,145,920 bytes) and int8 with
0.9993 cosine fidelity to its full-precision self.

The obvious question is whether a student trained on code is any use on ordinary prose.
It is:

| endpoint | dims | p@1 | doc@3 | MRR |
|---|---|---|---|---|
| `aither-code-embed-0.6b` | 1024 | **0.893** | 1.000 | **0.940** |
| Qwen3-Embedding-0.6B | 1024 | 0.821 | 0.964 | 0.900 |
| nomic-embed-text-v1.5 | 768 | 0.821 | 1.000 | 0.899 |
| all-MiniLM-L6-v2 | 384 | 0.750 | 0.964 | 0.851 |

**Method.** One real customer workspace: 26 content chunks from six documents (a firm
profile, a home page, four project profiles). 28 queries — 7 taken from questions users
had actually asked of that corpus, 21 written against the document text for even
coverage of every file. All four models served identically (llama.cpp, GGUF Q8_0, GPU);
each was given the query convention its authors document — nomic its
`search_query:`/`search_document:` prefixes, Qwen3 and the student their instruct-style
preambles — and the raw forms were measured too.

Metrics are **document-level**: a query counts as a hit when its gold document is what
the top-ranked chunk came from, so a corpus's chunking cannot flatter or punish a model.

**Findings worth carrying:**

- The code-search query prefix changes *nothing* on prose — identical rankings with and
  without it. The distillation transferred retrieval geometry; the instruction wording
  is a code-corpus artifact.
- Truncating the student to 256 dimensions — the form a volunteer-compute plane stores
  and verifies — costs real questions: p@1 0.714 → 0.429. Narrow vectors are the right
  shape for *verifying that two peers agree* and the wrong shape for *ranking*.
- Throughput is the trade, and it points the other way: nomic embeds ~1.6× faster on the
  same GPU (91.6 vs 58.6 chunks/s) at a quarter of the parameters. Single-query latency
  is 11.6 ms vs 17.1 ms — both are noise next to an LLM turn, but a bulk re-index is
  where the difference is real.

**Caveat, stated plainly.** 28 queries over 6 documents makes a two-query gap
directional, not significant, and this is one corpus. What the table supports is narrow:
*a student distilled on code transferred to ordinary prose well enough to beat the
general-purpose embedders compared here, on this corpus, including the fallback model
that was actually serving it.* It does not say the student is universally better.

**Reproduce it on your own corpus** with the `compare` stage of
[awembed](https://github.com/Aitherium/awembed) — black-box over OpenAI-shaped
`/v1/embeddings`, so your student, the teacher, and any third-party model are measured
by the same code on the same rows:

```bash
awembed compare --corpus docs.jsonl --queries queries.jsonl \
  --endpoint student=http://127.0.0.1:18101 \
  --endpoint nomic=http://127.0.0.1:18103 \
  --qprefix nomic="search_query: " --dprefix nomic="search_document: " \
  --out compare.json
```

`docs.jsonl` is one row per chunk (`{"id", "text"}`); `queries.jsonl` is
`{"query", "gold"}` where `gold` names a corpus id. The stage refuses to score — rather
than printing zeros — when an endpoint is unreachable, returns the wrong number of
vectors, or when the gold column names nothing in the corpus.

---

## TurboQuant kernels

Kernel-level benchmarks (compression ratio, reconstruction fidelity, and the
information-theoretic gap) live in [`notebooks/`](notebooks/) and are reproducible from
the classes in [`turboquant/`](turboquant/) — see **Research / benchmarking** in the
[README](README.md#research--benchmarking).
