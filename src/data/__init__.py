"""Data loading and validation for Alpamayo distillation."""

from src.data.teacher_dump import (
    CoCTrace,
    ConditioningMeta,
    TeacherClipManifest,
    TeacherDumpDataset,
    TeacherDumpExample,
    collate_teacher_examples,
    validate_teacher_clip,
)

__all__ = [
    "CoCTrace",
    "ConditioningMeta",
    "TeacherClipManifest",
    "TeacherDumpDataset",
    "TeacherDumpExample",
    "collate_teacher_examples",
    "validate_teacher_clip",
]
