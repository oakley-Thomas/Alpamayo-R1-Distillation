"""Teacher dump export helpers and Alpamayo runtime entrypoint.

The pure helpers in this module are intentionally testable without importing
the Alpamayo submodule. Runtime functions import Alpamayo lazily because they
require the 10B teacher model, CUDA, and PhysicalAI dataset access.
"""

from __future__ import annotations

import copy
import importlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image

from src.models.teacher_iface import AlpamayoConditioningRecord

DEFAULT_MODEL_NAME = "nvidia/Alpamayo-1.5-10B"
DEFAULT_TOP_K = 32
DEFAULT_NUM_TRAJ_SAMPLES = 16
DEFAULT_TRAJ_SAMPLE_BATCH_SIZE = 1
DEFAULT_MAX_GENERATION_LENGTH = 256
TEACHER_ACTION_MODULE_NAMES = (
    "expert",
    "diffusion",
    "action_in_proj",
    "action_out_proj",
    "action_space",
)


class TeacherDumpExportError(RuntimeError):
    """Raised when teacher dump export cannot produce the required contract."""


@dataclass(frozen=True)
class TopKTrace:
    """Top-k teacher logits aligned to generated token IDs."""

    token_ids: list[int]
    top_k_token_ids: list[list[int]]
    top_k_logits: list[list[float]]


@dataclass(frozen=True)
class ExportedRollout:
    """Raw tensors produced by one teacher rollout for dump serialization."""

    sequences: torch.Tensor
    logits: tuple[torch.Tensor, ...]
    prompt_cache: Any
    prefill_seq_len: int
    offset: torch.Tensor
    rope_deltas: torch.Tensor
    attention_mask_shape: tuple[int, ...]
    pred_xyz: torch.Tensor
    denoising_x_t: np.ndarray
    denoising_t: np.ndarray
    denoising_v_pred: np.ndarray


@dataclass(frozen=True)
class ExportTeacherDumpConfig:
    """Configuration for exporting Alpamayo Stage 1 teacher dump clips."""

    clip_ids: Path
    output_root: Path
    model_name: str = DEFAULT_MODEL_NAME
    num_traj_samples: int = DEFAULT_NUM_TRAJ_SAMPLES
    traj_sample_batch_size: int = DEFAULT_TRAJ_SAMPLE_BATCH_SIZE
    top_k: int = DEFAULT_TOP_K
    max_generation_length: int = DEFAULT_MAX_GENERATION_LENGTH
    temperature: float = 0.6
    top_p: float = 0.98
    t0_us: int = 5_100_000
    include_kv_cache: bool = False
    capture_denoising: bool = True
    require_stage3_fields: bool = True
    overwrite: bool = False


def _import_attr(module_name: str, attr_name: str) -> Any:
    """Import an attribute from a runtime-only dependency."""
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def read_clip_ids(path: str | Path) -> list[str]:
    """Read clip IDs from a JSON list or object with a ``clip_ids`` list."""
    split_path = Path(path)
    with split_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        clip_ids = cast(list[object], data)
    elif isinstance(data, dict):
        data_obj = cast(dict[str, object], data)
        raw_clip_ids = data_obj.get("clip_ids")
        clip_ids = cast(list[object], raw_clip_ids) if isinstance(raw_clip_ids, list) else None
    else:
        clip_ids = None
    if clip_ids is None or not all(isinstance(item, str) for item in clip_ids):
        raise TeacherDumpExportError(f"{split_path} must contain a list of clip ID strings")
    return [item for item in clip_ids if isinstance(item, str)]


def extract_primary_top_k(
    generated_token_ids: torch.Tensor,
    generation_logits: tuple[torch.Tensor, ...] | list[torch.Tensor],
    top_k: int,
    pad_token_id: int | None = None,
) -> TopKTrace:
    """Extract top-k logits for the primary generated sequence.

    Args:
        generated_token_ids: Generated tokens, shape (N, L) or (L,).
        generation_logits: One logits tensor per generated step, each shape (N, V).
        top_k: Number of logits to store per token.
        pad_token_id: Optional padding token ID to drop from the trace.

    Returns:
        Token IDs plus top-k token IDs/logits for the primary sequence.
    """
    if generated_token_ids.ndim == 2:
        primary_tokens = generated_token_ids[0]
    elif generated_token_ids.ndim == 1:
        primary_tokens = generated_token_ids
    else:
        raise ValueError("generated_token_ids must have shape (N, L) or (L,)")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not generation_logits:
        raise ValueError("generation_logits must contain at least one step")

    length = min(primary_tokens.shape[0], len(generation_logits))
    token_ids: list[int] = []
    top_k_token_ids: list[list[int]] = []
    top_k_logits: list[list[float]] = []

    for step_idx in range(length):
        token_id = int(primary_tokens[step_idx].item())
        if pad_token_id is not None and token_id == pad_token_id:
            continue
        step_logits = generation_logits[step_idx]
        if step_logits.ndim != 2:
            raise ValueError("each logits step must have shape (N, V)")
        values, indices = torch.topk(step_logits[0].detach().float().cpu(), k=top_k)
        index_array = indices.to(dtype=torch.int64).cpu().numpy()
        value_array = values.cpu().numpy()
        token_ids.append(token_id)
        top_k_token_ids.append([int(item) for item in index_array])
        top_k_logits.append([float(item) for item in value_array])

    if not token_ids:
        raise TeacherDumpExportError("primary generated sequence contains no non-padding tokens")
    return TopKTrace(
        token_ids=token_ids, top_k_token_ids=top_k_token_ids, top_k_logits=top_k_logits
    )


def select_hidden_states_until_offset(hidden_states: torch.Tensor, offset: int) -> np.ndarray:
    """Select fp16 teacher hidden states up to Alpamayo's expert-conditioning offset.

    Args:
        hidden_states: Final-layer hidden states, shape (B, L, D_h) or (L, D_h).
        offset: Position immediately after ``<|traj_future_start|>``.

    Returns:
        Hidden states for the primary sequence, shape (T, D_h), dtype fp16.
    """
    if hidden_states.ndim == 3:
        primary_hidden = hidden_states[0]
    elif hidden_states.ndim == 2:
        primary_hidden = hidden_states
    else:
        raise ValueError("hidden_states must have shape (B, L, D_h) or (L, D_h)")
    if offset <= 0 or offset > primary_hidden.shape[0]:
        raise ValueError(
            f"offset must be in [1, {primary_hidden.shape[0]}], got {offset}",
        )
    return primary_hidden[:offset].detach().cpu().to(torch.float16).numpy()


def conditioning_record_to_json(record: AlpamayoConditioningRecord) -> dict[str, Any]:
    """Return a JSON-serializable conditioning metadata dictionary."""
    return record.to_json_dict()


def ensure_denoising_available(
    capture_denoising: bool,
    require_stage3_fields: bool,
) -> None:
    """Reject Stage 3-ready export requests that omit denoising trajectories."""
    if require_stage3_fields and not capture_denoising:
        raise TeacherDumpExportError(
            "Stage 3-ready teacher dumps require --capture-denoising; "
            "use --stage2-only to omit denoising fields.",
        )


def trajectory_sample_batch_sizes(num_traj_samples: int, batch_size: int) -> tuple[int, ...]:
    """Split requested teacher trajectory samples into rollout batch sizes.

    Args:
        num_traj_samples: Total number of trajectory samples to export.
        batch_size: Maximum number of VLM return sequences per rollout.

    Returns:
        Tuple of positive rollout sizes whose sum is ``num_traj_samples``.
    """
    if num_traj_samples <= 0:
        raise TeacherDumpExportError("num_traj_samples must be positive")
    if batch_size <= 0:
        raise TeacherDumpExportError("traj_sample_batch_size must be positive")

    sizes: list[int] = []
    remaining = num_traj_samples
    while remaining > 0:
        current = min(batch_size, remaining)
        sizes.append(current)
        remaining -= current
    return tuple(sizes)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _save_frames(raw_data: dict[str, Any], frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = raw_data["image_frames"].detach().cpu()
    camera_indices = raw_data["camera_indices"].detach().cpu()
    for camera_idx in range(frames.shape[0]):
        view = int(camera_indices[camera_idx].item())
        for frame_idx in range(frames.shape[1]):
            frame = frames[camera_idx, frame_idx].permute(1, 2, 0).numpy()
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            Image.fromarray(frame).save(frames_dir / f"frame_{frame_idx:04d}_view{view}.jpg")


def _build_model_inputs(
    model: Any, raw_data: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    helper = _import_attr("alpamayo1_5", "helper")

    processor = helper.get_processor(model.tokenizer)
    messages = helper.create_message(
        raw_data["image_frames"].flatten(0, 1),
        camera_indices=raw_data["camera_indices"],
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": raw_data["ego_history_xyz"],
        "ego_history_rot": raw_data["ego_history_rot"],
    }
    return helper.to_device(model_inputs, device)


def _extract_coc_text(tokenizer: Any, generated_tokens: torch.Tensor) -> str:
    fallback = str(tokenizer.decode(generated_tokens[0], skip_special_tokens=False))
    try:
        extract_text_tokens = _import_attr("alpamayo1_5.models.token_utils", "extract_text_tokens")

        extracted = extract_text_tokens(tokenizer, generated_tokens[:1])
        text = extracted.get("cot", [""])[0]
        if text:
            return str(text)
    except (ImportError, KeyError, IndexError, TypeError):
        return fallback
    return fallback


def replay_hidden_states_for_export(
    model: Any,
    tokenized_data: dict[str, Any],
    sequence: torch.Tensor,
    offset: int,
) -> np.ndarray:
    """Replay the teacher VLM base model and return final-layer conditioning states.

    Args:
        model: Alpamayo teacher model exposing ``vlm.model``.
        tokenized_data: Processor outputs containing image/video tensors for the prompt.
        sequence: Full prompt plus generated sequence, shape (L,).
        offset: Position immediately after ``<|traj_future_start|>``.

    Returns:
        Hidden states up to ``offset``, shape (T, D_h), dtype fp16.
    """
    vlm_base = getattr(model.vlm, "model", None)
    if vlm_base is None:
        raise TeacherDumpExportError("Teacher VLM does not expose a base model for hidden replay")
    replay_kwargs: dict[str, Any] = {
        "input_ids": sequence.unsqueeze(0),
        "attention_mask": torch.ones_like(sequence, dtype=torch.long).unsqueeze(0),
        "output_hidden_states": False,
        "return_dict": True,
        "use_cache": False,
    }
    for key in (
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "mm_token_type_ids",
    ):
        if key in tokenized_data:
            replay_kwargs[key] = tokenized_data[key]
    with torch.inference_mode():
        outputs = vlm_base(**replay_kwargs)
    hidden_states = (
        outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
    )
    return select_hidden_states_until_offset(hidden_states, offset)


def offload_teacher_action_modules_for_replay(model: Any) -> tuple[str, ...]:
    """Move teacher action-side modules to CPU before VLM hidden-state replay.

    Args:
        model: Alpamayo teacher model.

    Returns:
        Names of modules that were moved and should be restored before another rollout.
    """
    moved: list[str] = []
    for name in TEACHER_ACTION_MODULE_NAMES:
        module = getattr(model, name, None)
        if isinstance(module, torch.nn.Module):
            module.to("cpu")
            moved.append(name)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tuple(moved)


def restore_teacher_action_modules_after_replay(
    model: Any,
    module_names: tuple[str, ...],
    device: torch.device,
) -> None:
    """Move teacher action-side modules back to the export device after replay.

    Args:
        model: Alpamayo teacher model.
        module_names: Module names returned by ``offload_teacher_action_modules_for_replay``.
        device: Device used for teacher rollout.
    """
    for name in module_names:
        module = getattr(model, name, None)
        if isinstance(module, torch.nn.Module):
            module.to(device)


def _trajectory_velocity_from_steps(trajectories: torch.Tensor, times: torch.Tensor) -> np.ndarray:
    if trajectories.shape[0] < 2:
        raise TeacherDumpExportError("Denoising capture produced fewer than two trajectory steps")
    dt = torch.diff(times.detach().cpu().float()).clamp_min(1e-6)
    velocity = torch.diff(trajectories.detach().cpu().float(), dim=0) / dt[:, None, None]
    velocity = torch.cat([velocity, velocity[-1:].clone()], dim=0)
    return velocity.numpy().astype(np.float32)


def _rollout_for_export(
    model: Any,
    model_inputs: dict[str, Any],
    config: ExportTeacherDumpConfig,
    device: torch.device,
    capture_logits: bool = True,
) -> ExportedRollout:
    from einops import rearrange, repeat
    from transformers import LogitsProcessorList, StoppingCriteriaList

    ExpertLogitsProcessor = _import_attr("alpamayo1_5.models.alpamayo1_5", "ExpertLogitsProcessor")
    StopAfterEOS = _import_attr("alpamayo1_5.models.token_utils", "StopAfterEOS")
    replace_padding_after_eos = _import_attr(
        "alpamayo1_5.models.token_utils", "replace_padding_after_eos"
    )
    to_special_token = _import_attr("alpamayo1_5.models.token_utils", "to_special_token")

    data = copy.deepcopy(model_inputs)
    n_samples_total = config.num_traj_samples
    ego_history_xyz = data["ego_history_xyz"]
    ego_history_rot = data["ego_history_rot"]
    tokenized_data = data["tokenized_data"]
    input_ids = tokenized_data.pop("input_ids")
    input_ids = model.fuse_traj_tokens(
        input_ids,
        {"ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot},
    )

    generation_config = copy.deepcopy(model.vlm.generation_config)
    generation_config.top_p = config.top_p
    generation_config.temperature = config.temperature
    generation_config.do_sample = True
    generation_config.num_return_sequences = config.num_traj_samples
    generation_config.max_new_tokens = config.max_generation_length
    generation_config.output_logits = capture_logits
    generation_config.return_dict_in_generate = True
    generation_config.pad_token_id = model.tokenizer.pad_token_id

    eos_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    vlm_outputs = model.vlm.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        stopping_criteria=StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)]),
        logits_processor=LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=model.config.traj_token_start_idx,
                    traj_vocab_size=model.config.traj_vocab_size,
                )
            ]
        ),
        **tokenized_data,
    )
    vlm_outputs.rope_deltas = model.vlm.model.rope_deltas
    vlm_outputs.sequences = replace_padding_after_eos(
        token_ids=vlm_outputs.sequences,
        eos_token_id=eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
    )

    prompt_cache = vlm_outputs.past_key_values
    prefill_seq_len = prompt_cache.get_seq_length()
    b_star = vlm_outputs.sequences.shape[0]
    n_diffusion_tokens = model.action_space.get_action_space_dims()[0]
    offset = model._find_eos_offset(vlm_outputs.sequences, eos_token_id, device)
    prefix_mask = tokenized_data.get("attention_mask")
    if prefix_mask is not None:
        prefix_mask = torch.repeat_interleave(prefix_mask, n_samples_total, dim=0)
    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset,
        rope_deltas=vlm_outputs.rope_deltas,
        kv_cache_seq_len=prefill_seq_len,
        n_diffusion_tokens=n_diffusion_tokens,
        b_star=b_star,
        device=device,
        prefix_mask=prefix_mask,
    )

    forward_kwargs: dict[str, Any] = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b_local = x.shape[0]
        future_token_embeds = model.action_in_proj(x, t)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(b_local, n_diffusion_tokens, -1)
        expert_out = model.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=prompt_cache,
            attention_mask=attention_mask,
            use_cache=True,
            **forward_kwargs,
        )
        prompt_cache.crop(prefill_seq_len)
        pred = model.action_out_proj(expert_out.last_hidden_state[:, -n_diffusion_tokens:]).view(
            -1,
            *model.action_space.get_action_space_dims(),
        )
        return pred

    total_batch = n_samples_total
    if config.capture_denoising:
        sampled_actions, time_steps = model.diffusion.sample(
            batch_size=total_batch,
            step_fn=step_fn,
            device=device,
            return_all_steps=True,
        )
        hist_xyz_rep = repeat(ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total)
        hist_rot_rep = repeat(ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total)
        flat_steps = sampled_actions.flatten(0, 1)
        hist_xyz_steps = repeat(hist_xyz_rep, "b ... -> (b s) ...", s=sampled_actions.shape[1])
        hist_rot_steps = repeat(hist_rot_rep, "b ... -> (b s) ...", s=sampled_actions.shape[1])
        step_xyz, _step_rot = model.action_space.action_to_traj(
            flat_steps, hist_xyz_steps, hist_rot_steps
        )
        step_xyz = rearrange(step_xyz, "(b s) ... -> b s ...", b=sampled_actions.shape[0])
        denoising_x_t = step_xyz[0].detach().cpu().float().numpy().astype(np.float32)
        denoising_t = time_steps.detach().cpu().float().numpy().astype(np.float32)
        denoising_v_pred = _trajectory_velocity_from_steps(step_xyz[0], time_steps)
        sampled_action = sampled_actions[:, -1]
    else:
        sampled_action = model.diffusion.sample(
            batch_size=total_batch,
            step_fn=step_fn,
            device=device,
            return_all_steps=False,
        )
        denoising_x_t = np.empty((0, 64, 3), dtype=np.float32)
        denoising_t = np.empty((0,), dtype=np.float32)
        denoising_v_pred = np.empty((0, 64, 3), dtype=np.float32)

    hist_xyz_rep = repeat(ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total)
    hist_rot_rep = repeat(ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total)
    pred_xyz, _pred_rot = model.action_space.action_to_traj(
        sampled_action, hist_xyz_rep, hist_rot_rep
    )

    return ExportedRollout(
        sequences=vlm_outputs.sequences,
        logits=tuple(vlm_outputs.logits) if capture_logits else (),
        prompt_cache=prompt_cache,
        prefill_seq_len=prefill_seq_len,
        offset=offset,
        rope_deltas=vlm_outputs.rope_deltas,
        attention_mask_shape=tuple(int(item) for item in attention_mask.shape),
        pred_xyz=pred_xyz,
        denoising_x_t=denoising_x_t,
        denoising_t=denoising_t,
        denoising_v_pred=denoising_v_pred,
    )


def _write_kv_cache(
    prompt_cache: Any, conditioning_dir: Path, include_kv_cache: bool
) -> tuple[str, ...]:
    if not include_kv_cache:
        return ()
    conditioning_dir.mkdir(parents=True, exist_ok=True)
    if not hasattr(prompt_cache, "to_legacy_cache"):
        raise TeacherDumpExportError(
            "Teacher KV cache cannot be serialized: missing to_legacy_cache"
        )
    filenames: list[str] = []
    for layer_idx, layer_cache in enumerate(prompt_cache.to_legacy_cache()):
        key, value = layer_cache
        filename = f"kv_layer_{layer_idx:03d}.npz"
        np.savez(
            conditioning_dir / filename,
            key=key[:1].detach().cpu().to(torch.float16).numpy(),
            value=value[:1].detach().cpu().to(torch.float16).numpy(),
        )
        filenames.append(filename)
    return tuple(filenames)


def export_teacher_clip(
    model: Any,
    clip_id: str,
    config: ExportTeacherDumpConfig,
    device: torch.device,
    avdi: Any | None = None,
) -> None:
    """Export one clip from Alpamayo teacher inference to the Stage 1 dump contract."""
    load_physical_aiavdataset = _import_attr(
        "alpamayo1_5.load_physical_aiavdataset", "load_physical_aiavdataset"
    )

    ensure_denoising_available(config.capture_denoising, config.require_stage3_fields)
    clip_dir = config.output_root / clip_id
    if clip_dir.exists() and not config.overwrite:
        raise TeacherDumpExportError(f"Clip {clip_id}: output directory already exists")
    clip_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_data = load_physical_aiavdataset(clip_id, t0_us=config.t0_us, avdi=avdi)
        model_inputs = _build_model_inputs(model, raw_data, device)
        tokenized_data = cast(dict[str, Any], model_inputs["tokenized_data"])
        prompt_input_ids = cast(torch.Tensor, tokenized_data["input_ids"])
        prompt_length = prompt_input_ids.shape[1]
        del prompt_input_ids

        trace: TopKTrace | None = None
        primary_text: str | None = None
        primary_offset: int | None = None
        sequence_for_replay: torch.Tensor | None = None
        denoising_x_t: np.ndarray | None = None
        denoising_t: np.ndarray | None = None
        denoising_v_pred: np.ndarray | None = None
        generated_seq_len: int | None = None
        prefill_seq_len: int | None = None
        rope_deltas: tuple[int, ...] | None = None
        attention_mask_shape: tuple[int, ...] | None = None
        kv_files: tuple[str, ...] = ()
        trajectory_chunks: list[np.ndarray] = []
        n_diffusion_tokens = int(model.action_space.get_action_space_dims()[0])

        sample_batches = trajectory_sample_batch_sizes(
            config.num_traj_samples,
            config.traj_sample_batch_size,
        )

        for batch_idx, sample_count in enumerate(sample_batches):
            is_primary_batch = batch_idx == 0
            batch_config = replace(
                config,
                num_traj_samples=sample_count,
                capture_denoising=config.capture_denoising and is_primary_batch,
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                rollout = _rollout_for_export(
                    model,
                    model_inputs,
                    batch_config,
                    device,
                    capture_logits=is_primary_batch,
                )

            trajectory_chunks.append(
                rollout.pred_xyz.detach().cpu().float().numpy().astype(np.float32)
            )

            if is_primary_batch:
                generated_tokens = rollout.sequences[:, prompt_length:]
                trace = extract_primary_top_k(
                    generated_tokens,
                    rollout.logits,
                    top_k=config.top_k,
                    pad_token_id=model.tokenizer.pad_token_id,
                )
                primary_text = _extract_coc_text(model.tokenizer, generated_tokens)
                primary_offset = int(rollout.offset[0].item())
                sequence_for_replay = rollout.sequences[0].detach().clone()
                denoising_x_t = rollout.denoising_x_t
                denoising_t = rollout.denoising_t
                denoising_v_pred = rollout.denoising_v_pred
                generated_seq_len = int(generated_tokens.shape[1])
                prefill_seq_len = rollout.prefill_seq_len
                rope_deltas = tuple(
                    int(item)
                    for item in rollout.rope_deltas.reshape(-1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int64)
                )
                attention_mask_shape = rollout.attention_mask_shape
                kv_files = _write_kv_cache(
                    rollout.prompt_cache,
                    clip_dir / "conditioning",
                    include_kv_cache=config.include_kv_cache,
                )
                del generated_tokens

            del rollout
            torch.cuda.empty_cache()

        if (
            trace is None
            or primary_text is None
            or primary_offset is None
            or sequence_for_replay is None
            or denoising_x_t is None
            or denoising_t is None
            or denoising_v_pred is None
            or generated_seq_len is None
            or prefill_seq_len is None
            or rope_deltas is None
            or attention_mask_shape is None
        ):
            raise TeacherDumpExportError("teacher rollout produced no primary batch")

        trajectories = np.concatenate(trajectory_chunks, axis=0)
        del model_inputs
        torch.cuda.empty_cache()
        offloaded_modules = offload_teacher_action_modules_for_replay(model)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden_states = replay_hidden_states_for_export(
                model,
                tokenized_data,
                sequence_for_replay,
                offset=primary_offset,
            )
        torch.cuda.empty_cache()
        restore_teacher_action_modules_after_replay(model, offloaded_modules, device)
    except Exception as exc:
        raise TeacherDumpExportError(f"Clip {clip_id}: export failed: {exc}") from exc

    _write_json(
        clip_dir / "meta.json",
        {
            "clip_id": clip_id,
            "num_frames": int(raw_data["image_frames"].shape[1]),
            "fps": 10,
            "camera_intrinsics": [],
            "source_url": "physical-ai-av",
            "t0_us": int(raw_data["t0_us"]),
        },
    )
    _save_frames(raw_data, clip_dir / "frames")
    _write_json(
        clip_dir / "coc_trace.json",
        [
            {
                "step_idx": 0,
                "text": primary_text,
                "token_ids": trace.token_ids,
                "top_k_token_ids": trace.top_k_token_ids,
                "top_k_logits": trace.top_k_logits,
            }
        ],
    )
    np.save(clip_dir / "hidden_states.npy", hidden_states)
    np.save(
        clip_dir / "trajectories.npy",
        trajectories,
    )
    np.savez(
        clip_dir / "denoising_traj.npz",
        x_t=denoising_x_t,
        t=denoising_t,
        v_pred=denoising_v_pred,
    )
    if config.include_kv_cache:
        record = AlpamayoConditioningRecord(
            traj_future_start_offset=primary_offset,
            prefill_seq_len=prefill_seq_len,
            rope_deltas=rope_deltas,
            attention_mask_shape=attention_mask_shape,
            generated_seq_len=generated_seq_len,
            n_diffusion_tokens=n_diffusion_tokens,
            kv_cache_files=kv_files,
        )
        _write_json(
            clip_dir / "conditioning" / "conditioning_meta.json",
            conditioning_record_to_json(record),
        )


def export_teacher_dump(config: ExportTeacherDumpConfig) -> None:
    """Export all requested clips with the real Alpamayo teacher model."""
    if not torch.cuda.is_available():
        raise TeacherDumpExportError("Teacher dump export requires CUDA")
    ensure_denoising_available(config.capture_denoising, config.require_stage3_fields)
    Alpamayo1_5 = _import_attr("alpamayo1_5.models.alpamayo1_5", "Alpamayo1_5")

    device = torch.device("cuda")
    clip_ids = read_clip_ids(config.clip_ids)
    config.output_root.mkdir(parents=True, exist_ok=True)
    model = Alpamayo1_5.from_pretrained(config.model_name, dtype=torch.bfloat16).to(device)
    model.eval()

    for clip_id in clip_ids:
        export_teacher_clip(model=model, clip_id=clip_id, config=config, device=device)
