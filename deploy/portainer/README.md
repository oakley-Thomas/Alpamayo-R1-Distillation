# Portainer Teacher Export

This stack runs the Alpamayo teacher as a one-shot GPU batch job and writes the
Stage 1 teacher dump to persistent Docker volumes. At container startup, the
image clones this project from GitHub, checks out the requested ref, initializes
the repo's `alpamayo1.5` submodule, and runs export from that checkout.

## Persistent Volumes

Create these external volumes in Portainer before deploying the stack:

```text
alpamayo_hf_cache
alpamayo_data
alpamayo_outputs
```

`alpamayo_hf_cache` is mounted at `/cache/huggingface`; Hugging Face model
weights and streamed PhysicalAI dataset files live there across container runs.
`alpamayo_data` is mounted at `/workspace/data`; it should contain
`splits/train.json`, `splits/val.json`, and `splits/test.json`, and receives the
teacher dump under `teacher_dump/`. `alpamayo_outputs` is reserved for later
training and evaluation artifacts.

You can override the volume names with Portainer stack environment variables:

```text
HF_CACHE_VOLUME=my_hf_cache
DATA_VOLUME=my_alpamayo_data
OUTPUTS_VOLUME=my_alpamayo_outputs
```

## Required Stack Variables

Set this in Portainer when deploying:

```text
HF_TOKEN=<your Hugging Face token>
```

The token must have accepted access to:

- `nvidia/Alpamayo-1.5-10B`
- `nvidia/PhysicalAI-Autonomous-Vehicles`

## Image

Build and push the image to a registry that your Portainer host can pull:

```bash
IMAGE_NAME=your-registry/alpamayo-distill:stage2 scripts/docker/build.sh
docker push your-registry/alpamayo-distill:stage2
```

Then set:

```text
IMAGE_NAME=your-registry/alpamayo-distill:stage2
```

If the image is already built locally on the Portainer host, you can leave the
default `alpamayo-distill:stage2`.

The image pulls the repo at runtime. Override these only when you need a
different fork, branch, tag, or commit:

```text
REPO_URL=https://github.com/oakley-Thomas/CSE676-Project.git
REPO_REF=VLM-Backbone
```

If the GitHub repo is private, set `GITHUB_TOKEN` to a token with read access.

## Running Exports

By default the stack exports every existing split file at:

```text
/workspace/data/splits/train.json
/workspace/data/splits/val.json
/workspace/data/splits/test.json
```

To export a single split, set:

```text
SPLITS=data/splits/val.json
```

To export several explicit split files, use a comma-separated list:

```text
SPLITS=data/splits/train.json,data/splits/val.json,data/splits/test.json
```

The job validates exported clips after finishing by default. Set
`VALIDATE_AFTER_EXPORT=false` to skip validation.

## Common Options

```text
NUM_TRAJ_SAMPLES=16
TOP_K=32
INCLUDE_KV_CACHE=true
CAPTURE_DENOISING=true
OVERWRITE=false
```

Keep `CAPTURE_DENOISING=true` for Stage 3-ready dumps. Set `STAGE2_ONLY=true`
only if you intentionally want to omit Stage 3 denoising fields.

## GPU Host Requirements

The Portainer host must have:

- NVIDIA driver installed
- NVIDIA Container Toolkit installed
- enough VRAM for Alpamayo-1.5-10B teacher inference

This stack is a batch job, not a web service, so it exposes no ports.
