# TIRA Submission Technical Log

**PAN 2026 — Voight-Kampff Generative AI Detection Task**
Team: `uev-japerdom` · Task: `generative-ai-authorship-verification-panclef-2026`
Period covered: 1 September 2026 – 4 September 2026
Document compiled: 4 September 2026

---

## Purpose of this document

This document is a chronological technical record of the engineering work required to
package four machine-learning models as reproducible Docker submissions and deploy them
to TIRA, the evaluation platform used by the PAN 2026 Voight-Kampff shared task. It is
intended as a reference for the corresponding Master's thesis.

Its scope is deliberately narrow. It records infrastructure and deployment work — the
repository layout, the continuous-integration pipeline, the containerisation of each
model, and the failures encountered along the way. It does **not** report classification
performance, since evaluation scores are produced by TIRA independently of the process
described here.

All statements below were verified against the repositories, their commit histories,
and the GitHub Actions run history via the GitHub REST API at the time of compilation.
Where the deployed code diverges from an earlier intended design, the code as deployed
is what is documented, and the divergence is noted explicitly.

---

## Section 1: Repository Structure

Work began on 1 September 2026 with a single repository, `pan26-voight-kampff`, holding
all four models as subdirectories. Its structure was:

```
pan26-voight-kampff/
├── .github/workflows/
│   └── upload-software-to-tira.yml
├── data/
│   ├── train.jsonl          # 23,707 labelled documents (~92 MB)
│   └── val.jsonl            #  3,589 labelled documents (~14 MB)
├── shared/
│   └── utils.py             # I/O and preprocessing shared by all models
├── tfidf_lr/                # TF-IDF + Logistic Regression
├── tfidf_svm/               # TF-IDF + calibrated LinearSVC
├── roberta_finetuned/       # Fine-tuned RoBERTa classifier
├── qwen3_lora/              # Qwen3-0.6B + QLoRA adapters
└── README.md
```

Each model subdirectory contained a `Dockerfile`, a `predict.py` entry point, and a
`requirements.txt`.

### Shared utilities

`shared/utils.py` centralises the operations common to all four models, which keeps the
input contract identical across submissions:

- `preprocess(text)` — NFC Unicode normalisation, removal of control characters, and
  whitespace collapsing.
- `resolve_jsonl_path(path)` — accepts either a file or a directory and, in the
  directory case, resolves the single `.jsonl`/`.jsonl.gz` file inside it. This matters
  because TIRA supplies a *dataset directory*, not a file path.
- `load_input(path)` / `open_jsonl(path)` — reads the input corpus, transparently
  handling gzip compression.
- `load_train(path)` — loads the training corpus and deduplicates it by MD5 hash of the
  case-folded, whitespace-normalised text.
- `write_predictions(output_dir, ids, scores)` — writes `predictions.jsonl` in the
  format TIRA expects, one JSON object per line with `id`, `label`, and `score` fields,
  scores rounded to four decimal places.

The initial repository history records the progression:

```
9766a83  Initial repository structure for PAN 2026 submission
0d4195c  Add all prediction models for PAN 2026 TIRA submission
13a8e0f  Point roberta_finetuned and qwen3_lora at eljosey40 HuggingFace repos
cc9a772  Fix roberta_finetuned HuggingFace repo name
3b96f78  Cast bfloat16 logits to float32 before numpy conversion in qwen3_lora
```

The last of these is worth noting as a representative low-level defect: NumPy cannot
convert a `bfloat16` tensor directly, so the Qwen3 inference path required an explicit
`.float()` cast on the logits before `.cpu().numpy()`.

---

## Section 2: TIRA Integration Setup

Submissions are delivered to TIRA through a GitHub Actions workflow,
`.github/workflows/upload-software-to-tira.yml`, triggered manually via
`workflow_dispatch` with a `directory` input identifying the submission root.

The workflow builds the Docker image, verifies the local TIRA installation, and uploads
the image as a code submission:

```yaml
- name: Build, test, and upload image
  run: |
    tira-cli login --token ${{ secrets.TIRA_CLIENT_TOKEN }}
    tira-cli verify-installation --task generative-ai-authorship-verification-panclef-2026 --team uev-japerdom
    tira-cli code-submission --path ${{ inputs.directory }} --task generative-ai-authorship-verification-panclef-2026 --dataset generative-ai-authorship-verification-panclef-2026/pan26-generative-ai-detection-smoke-test-20260330-training
```

Reaching a working configuration took **eight consecutive failed runs** on
`pan26-voight-kampff`, between `2026-09-01T14:14:46Z` and `2026-09-02T07:36:19Z`. Four
distinct problems were diagnosed and fixed in sequence.

### 2.1 Python 3.8 incompatibility with `tira-cli`

The runner's default interpreter was Python 3.8. The `tira-cli` package uses PEP 585
built-in generic annotations (`list[str]` rather than `typing.List[str]`), which are a
syntax error before Python 3.9. Installation failed at import time.

Fixed by pinning the interpreter explicitly (commit `d2fb569`):

```yaml
- uses: actions/setup-python@v5
  with:
    python-version: '3.10'
```

### 2.2 Node.js 20 deprecation warnings

The workflow initially used v1/v2-era actions running on a deprecated Node.js runtime,
which GitHub had scheduled for removal. All actions were upgraded (commits `8bb94e6`,
`7ad97a5`) to `actions/checkout@v4`, `actions/setup-python@v5`,
`docker/setup-qemu-action@v3`, and `docker/setup-buildx-action@v3`.

### 2.3 No Dockerfile at the submission root

`tira-cli code-submission` requires a `Dockerfile` at the root of the directory it is
given. With four models living in subdirectories of one repository, no single value of
`--path` satisfied this for more than one model at a time. This constraint drove the
repository restructuring described in Section 5.2.

### 2.4 Incorrect team identifier

`tira-cli verify-installation` was initially invoked with `--team pan26-voight-kampff`,
the repository name, which is not a TIRA team. The correct identifier, `uev-japerdom`,
became available only after the TIRA group invitation was accepted, and was applied in
commit `110d26e` ("Correct TIRA team name to uev-japerdom").

### 2.5 Final working configuration

Python 3.10; `actions/checkout@v4` with `lfs: true`; `actions/setup-python@v5`;
`docker/setup-qemu-action@v3`; `docker/setup-buildx-action@v3`; team `uev-japerdom`;
45-minute job timeout.

The `lfs: true` parameter is essential and was itself the subject of a later fix
(`964c834`, "Fetch git-lfs objects during checkout"). Without it, `actions/checkout`
writes Git LFS *pointer files* — small text stubs — into the working tree instead of the
model weights. The Docker build then succeeds while embedding a few hundred bytes of
text where half a gigabyte of tensors should be, and the failure only surfaces at
inference time inside TIRA.

---

## Section 3: Model Submission — Baselines

The two baseline models are classical sparse-feature classifiers that require no GPU and
carry no pretrained weights. Both are trained *at inference time*, inside the TIRA
container, from the `train.jsonl` corpus vendored into the image.

| Model | Repository | Method |
|---|---|---|
| `tfidf_lr` | `Eljoseygamer/tfidf_lr` | TF-IDF features → Logistic Regression |
| `tfidf_svm` | `Eljoseygamer/tfidf_svm` | TF-IDF features → `LinearSVC`, probability-calibrated |

`LinearSVC` produces uncalibrated decision-function values rather than probabilities,
which is incompatible with the task's requirement for a score in `[0, 1]`. Calibration
wraps the classifier so its output can be interpreted as a probability.

### Training-data path resolution

The defect encountered here was a working-directory dependency. `train.jsonl` was
referenced relative to the process working directory, which differs between local
execution from the repository root, execution from within a model subdirectory, and
execution inside the container, where TIRA sets its own working directory.

The resolution deployed is to anchor the path to the source file's own location rather
than to the working directory:

```python
train_path = os.path.join(os.path.dirname(__file__), 'data', 'train.jsonl')
```

Combined with vendoring a copy of `data/` into each baseline repository (commit
`ba198f8`, "Make each model directory self-contained for TIRA submission"), this makes
the path correct under every invocation context.

> **Note on the historical record.** An intermediate iteration of this fix used
> try-several-candidate-paths fallback logic. That approach is *not* what is deployed:
> the code in both baseline repositories resolves the path in a single `__file__`-anchored
> expression, with no fallback branch. The `__file__` anchor is the more robust of the
> two, since it cannot silently select the wrong corpus.

### Outcome

Both baselines were submitted successfully on 2 September 2026.

| Repository | Runs | First success |
|---|---|---|
| `tfidf_lr` | 3 (2 failed, 1 succeeded) | `2026-09-02T08:13:40Z` |
| `tfidf_svm` | 1 (succeeded first attempt) | `2026-09-02T08:18:38Z` |

`tfidf_svm` succeeding on its first attempt is a direct consequence of the workflow
having been debugged against `tfidf_lr` immediately beforehand.

---

## Section 4: Model Submission — GPU Models

The two neural models carry pretrained weights and are the reason GPU support in TIRA
matters. Both embed their weights in the image and run fully offline.

### 4.1 RoBERTa fine-tuned

Repository: `Eljoseygamer/roberta_finetuned`

The fine-tuned classifier is loaded with `RobertaForSequenceClassification` from a
`model/` directory committed to the repository through Git LFS. The weights file
`model/model.safetensors` is approximately 480 MB (498,612,824 bytes), accompanied by
the tokenizer vocabulary and configuration.

Inference configuration: batch size 32, maximum sequence length 512 tokens, and a
softmax over the two output logits, taking index 1 as the machine-generated score.

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(device)
```

`local_files_only=True` is required throughout: TIRA executes containers without network
access, and without this flag `transformers` attempts to contact the HuggingFace Hub and
fails.

Submission succeeded on `2026-09-02T09:38:46Z`, after three failed attempts
(`08:34:29Z`, `08:57:33Z`, `09:33:21Z`).

### 4.2 Qwen3-0.6B + QLoRA

Repository: `Eljoseygamer/qwen3_lora`

This submission embeds two artefacts:

- `base_model/` — Qwen3-0.6B, approximately 1.5 GB (1,503,300,328 bytes), via Git LFS.
- `adapters/` — the trained LoRA adapters, approximately 40 MB
  (`adapter_model.safetensors`, 393 tensors).

The base model is quantised to 4 bits at load time to reduce memory footprint:

```python
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```

Two implementation details follow from quantisation. First, `device_map='auto'` is used
instead of an explicit `.to(device)` call, because a 4-bit quantised model cannot be
moved between devices after construction the way an ordinary model can. Second, since
`device_map` decides placement itself, each batch must be moved to wherever the model
actually resides rather than to a device name chosen in advance:

```python
enc = {k: v.to(model.device) for k, v in enc.items()}
```

Inference uses batch size 16 — half the RoBERTa batch size, reflecting the larger model
— and the same 512-token limit. The tokenizer has no pad token by default, so
`tokenizer.pad_token` is set to `tokenizer.eos_token` and `model.config.pad_token_id`
updated to match.

Submission succeeded on `2026-09-02T10:29:08Z`, after one failed attempt (`09:42:02Z`).

### 4.3 The classification head: `score.weight`

A sequence-classification head does not exist in the Qwen3-0.6B base checkpoint, which
is a causal language model. `AutoModelForSequenceClassification.from_pretrained(...,
num_labels=2)` therefore constructs a **randomly initialised** `score` layer. If that
random layer survives into inference, the model's outputs are noise regardless of how
well the LoRA adapters were trained. This manifests during development as a
`score.weight` MISSING warning when loading the adapters.

The deployed solution is to have PEFT persist the head inside the adapter itself, via
the `modules_to_save` field of the adapter configuration:

```json
"modules_to_save": ["classifier", "score"],
"task_type": "SEQ_CLS"
```

Modules listed in `modules_to_save` are stored in full — not as low-rank deltas — and
restored by `PeftModel.from_pretrained`. Inspection of the adapter file confirms the
mechanism works as intended: `adapter_model.safetensors` contains the tensor
`base_model.model.score.weight` among its 393 entries. Consequently the single call

```python
model = PeftModel.from_pretrained(model, ADAPTER_DIR, is_trainable=False, local_files_only=True)
```

restores both the LoRA deltas and the trained classification head, and no separate
loading step is required.

> **Note on the historical record.** The repository also contains
> `adapters/classification_head.pt` (9,917 bytes), an artefact of an earlier approach in
> which the head was saved and re-loaded manually after `PeftModel.from_pretrained`.
> That approach was superseded by the `modules_to_save` mechanism described above.
> **The deployed `predict.py` does not read `classification_head.pt`**; the file is a
> harmless leftover. This is stated explicitly because the two mechanisms are easy to
> confuse, and only one of them is actually in effect.

### 4.4 Migration to a CUDA base image

Both GPU models were originally built on `python:3.10-slim`. That image contains no CUDA
runtime libraries, so `torch.cuda.is_available()` returns `False` inside the container
irrespective of the hardware TIRA provides. Both models therefore ran on CPU during the
2 September submissions, and the `device` selection logic silently fell back:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'   # always 'cpu' on python:3.10-slim
```

On 4 September 2026 both Dockerfiles were migrated to the CUDA base image used by the
official PAN 2026 baselines, `nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04`. The
deployed Dockerfile for `roberta_finetuned` is:

```dockerfile
FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

RUN set -x \
    && apt update \
    && apt install -y python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip config set global.break-system-packages true \
    && python3 -m pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY model/ model/
COPY predict.py .
CMD ["python3", "predict.py"]
```

`qwen3_lora` uses an identical Dockerfile except that it copies `base_model/` and
`adapters/` in place of `model/`.

Three changes deserve comment.

**Python must be installed explicitly.** The NVIDIA CUDA images are built on bare Ubuntu
and ship no Python interpreter, unlike `python:3.10-slim`. Ubuntu 24.04 provides Python
3.12 as `python3`; note that it provides no `python` alias, which is why the command form
is `python3 predict.py`.

**`break-system-packages` is required.** Ubuntu 24.04 marks its system Python as
externally managed under PEP 668, and `pip install` refuses to write into it. Setting
`python3 -m pip config set global.break-system-packages true` before installing lifts
that restriction. This mirrors the official PAN 2026 baseline Dockerfile.

**`ENTRYPOINT` was replaced by `CMD`.** The previous form was:

```dockerfile
ENTRYPOINT ["python", "predict.py", "$inputDataset", "$outputDir"]
```

This is exec form, in which no shell is involved and `$inputDataset` and `$outputDir`
are therefore **not expanded** — they were passed to `predict.py` as the literal
seven- and ten-character strings `$inputDataset` and `$outputDir`. The submissions
nonetheless worked, because `predict.py` reads the environment variables before
consulting `sys.argv` (Section 5.5); the literal arguments were simply never used.
Replacing the line with `CMD ["python3", "predict.py"]` removes the misleading
construct.

At the time of writing, the workflow runs for this migration are still executing:

- `roberta_finetuned` — run `33862326863`, dispatched `2026-09-04T10:15:07Z`, status *in progress*
- `qwen3_lora` — run `33862267146`, dispatched `2026-09-04T10:14:24Z`, status *in progress*

**Their outcome is therefore not yet part of this record.** The four successful
submissions reported in Sections 3 and 4 are those of 2 September 2026, which ran on
CPU. This document should be updated once the 4 September runs complete.

---

## Section 5: Key Technical Decisions

### 5.1 Weights embedded in the image rather than downloaded at runtime

TIRA executes submissions in a sandboxed environment with no network access. Any
approach that resolves a model identifier at runtime — for example
`from_pretrained('eljosey40/roberta-finetuned-pan26-voightkampff')` — fails there, even
though it works during local development, which makes it a particularly awkward class of
bug: it cannot be reproduced except by explicitly disabling networking.

The consequence is that roughly 2 GB of tensors had to be committed to Git. Git LFS was
used for all weight files, declared per repository in `.gitattributes`:

```
base_model/*.safetensors filter=lfs diff=lfs merge=lfs -text
adapters/*.safetensors   filter=lfs diff=lfs merge=lfs -text
adapters/*.pt            filter=lfs diff=lfs merge=lfs -text
```

Two safeguards make the offline requirement explicit rather than incidental:
`local_files_only=True` on every `from_pretrained` call, and paths pointing at
directories inside the image rather than at Hub identifiers.

### 5.2 One repository per model

The decision to split into four independent repositories was forced by the constraint in
Section 2.3: `tira-cli code-submission` requires a `Dockerfile` at the root of the
submitted path. Restructuring (commit `ba198f8`) gave each model its own repository
containing a root `Dockerfile`, its own copy of `shared/utils.py`, its own weights or
training data, and its own copy of the workflow.

The cost is duplication — `shared/utils.py` now exists in five places, and changes must
be propagated by hand. This is a genuine maintenance liability, accepted because the
platform constraint left no alternative. `pan26-voight-kampff` was retained as the
umbrella repository holding the training data, the shared source of truth for
`utils.py`, and this log.

### 5.3 CUDA base image

Covered in detail in Section 4.4. The decision was to follow the official PAN 2026
baseline rather than construct a base image independently, on the grounds that matching
the reference configuration minimises the risk of a driver or toolkit mismatch on
TIRA's hardware, which cannot be tested locally.

One residual risk should be recorded. The `requirements.txt` files pin only lower bounds
(`torch>=2.0`, `transformers>=4.51`). Local verification of the migrated image resolved
these to `torch 2.14.0+cu130` and `transformers 5.16.1` — a torch built against CUDA 13.0
running on a CUDA 12.8 base image, and a `transformers` major version ahead of the stated
minimum. The torch wheel bundles its own CUDA runtime, so the version skew is tolerated,
but it requires an NVIDIA driver of at least the 580 series on the execution host. Since
the resolution is performed afresh on each build, two builds of the same commit may not
produce the same dependency set. Pinning exact versions would remove both risks.

### 5.4 Restoring the classification head through `modules_to_save`

Covered in Section 4.3. The decision was to let PEFT own the full lifecycle of the
classification head rather than manage it as a side artefact, which keeps adapter
loading a single atomic operation and removes the possibility of loading LoRA weights
while forgetting the head.

### 5.5 Environment variables for input and output paths

TIRA communicates dataset and output locations through the environment variables
`inputDataset` and `outputDir`, not through command-line arguments. All four `predict.py`
scripts read them with a fallback chain that keeps local testing convenient:

```python
input_dir  = os.environ.get('inputDataset') or (sys.argv[1] if len(sys.argv) > 1 else '/tira-data/input')
output_dir = os.environ.get('outputDir')    or (sys.argv[2] if len(sys.argv) > 2 else '/tira-data/output')
```

The precedence — environment first, then arguments, then a TIRA-conventional default —
means the same script runs unmodified under TIRA, under a local `docker run` with
explicit arguments, and directly from a shell. As noted in Section 4.4, this precedence
also masked the unexpanded-variable defect in the old `ENTRYPOINT`.

`inputDataset` points to a *directory*; `resolve_jsonl_path` in `shared/utils.py`
locates the single `.jsonl` file within it.

---

## Section 6: Final Repository URLs

**Model repositories**

- `tfidf_lr` — https://github.com/Eljoseygamer/tfidf_lr
- `tfidf_svm` — https://github.com/Eljoseygamer/tfidf_svm
- `roberta_finetuned` — https://github.com/Eljoseygamer/roberta_finetuned
- `qwen3_lora` — https://github.com/Eljoseygamer/qwen3_lora

**Umbrella repository**

- `pan26-voight-kampff` — https://github.com/Eljoseygamer/pan26-voight-kampff

---

## Consolidated timeline

| Timestamp (UTC) | Repository | Event |
|---|---|---|
| 2026-09-01 14:14 – 14:42 | `pan26-voight-kampff` | 6 failed runs — Python 3.8, deprecated actions |
| 2026-09-02 07:32 – 07:36 | `pan26-voight-kampff` | 2 failed runs — no Dockerfile at root |
| 2026-09-02 — | all | Restructured into four independent repositories (`ba198f8`) |
| 2026-09-02 07:54, 08:08 | `tfidf_lr` | 2 failed runs |
| 2026-09-02 08:13:40 | `tfidf_lr` | **First successful submission** |
| 2026-09-02 08:18:38 | `tfidf_svm` | **Successful submission**, first attempt |
| 2026-09-02 08:34 – 09:33 | `roberta_finetuned` | 3 failed runs — incl. LFS pointer files |
| 2026-09-02 09:38:46 | `roberta_finetuned` | **Successful submission** |
| 2026-09-02 09:42 | `qwen3_lora` | 1 failed run |
| 2026-09-02 10:29:08 | `qwen3_lora` | **Successful submission** |
| 2026-09-04 10:14:24 | `qwen3_lora` | CUDA migration dispatched — *in progress* |
| 2026-09-04 10:15:07 | `roberta_finetuned` | CUDA migration dispatched — *in progress* |

Aggregate: 22 workflow runs, of which 4 produced the successful submissions of
2 September and 2 were still executing at the time of writing.

---

## Conclusion

Four models were packaged as self-contained Docker submissions and deployed to TIRA for
the PAN 2026 Voight-Kampff task between 1 and 4 September 2026. All four — `tfidf_lr`,
`tfidf_svm`, `roberta_finetuned`, and `qwen3_lora` — were submitted successfully on
2 September 2026 and executed on the task's evaluation dataset.

The engineering effort was dominated not by modelling but by the constraints of the
evaluation platform, and the failures cluster into three groups.

The first is **environment mismatch**: an interpreter too old for the client library, a
base image without CUDA, a base image without Python, and a distribution that refuses
`pip install` into its system interpreter. Each was individually trivial and each cost a
failed run to diagnose, because none could be observed locally.

The second is **the sandbox boundary**. TIRA's lack of network access invalidates the
normal HuggingFace workflow of resolving a model by name. Every weight had to be
committed through Git LFS and every load call marked `local_files_only=True`. The
associated failure mode is unusually treacherous: a checkout without `lfs: true` yields
pointer files, and the image builds cleanly with text stubs where the tensors should be,
failing only at inference.

The third is **silent fallbacks**. Two defects survived precisely because the system kept
working. `torch.cuda.is_available()` returning `False` on a CPU-only base image degraded
to CPU inference rather than raising, so the 2 September submissions ran without GPU
acceleration and reported success. An `ENTRYPOINT` in exec form passed the literal
strings `$inputDataset` and `$outputDir` for two days without consequence, because the
environment-variable path took precedence. Both illustrate a general point worth
carrying into the thesis: in a pipeline built from graceful degradations, a component
can fail completely while the pipeline reports success.

Two matters remain open at the time of writing. The CUDA migration runs dispatched on
4 September had not completed, so whether GPU acceleration is in fact obtained on TIRA's
hardware is unverified. And the dependency specifications remain unpinned lower bounds,
which makes builds non-reproducible across time — a property at odds with the
reproducibility guarantees that motivate the use of TIRA in the first place. Both should
be resolved before these results are cited as final.
