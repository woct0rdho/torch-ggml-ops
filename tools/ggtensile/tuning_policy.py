"""Shared typed policies for implemented GGTensile tuning mechanisms."""

from enum import Enum


class PipelineStage(str, Enum):
    SingleStage = "SingleStage"
    DoubleStage = "DoubleStage"


class LdsBuffering(str, Enum):
    Single = "Single"
    Double = "Double"


class ClusterLocalRead(str, Enum):
    Disabled = "Disabled"
    Enabled = "Enabled"


class GlobalReadSchedule(str, Enum):
    Default = "Default"
    Interleaved = "Interleaved"
    WeightThenActivation = "WeightThenActivation"


class LocalWriteSchedule(str, Enum):
    Default = "Default"
    Grouped = "Grouped"
    Split = "Split"


class IterationSchedule(str, Enum):
    DependencyOrdered = "DependencyOrdered"
    ExplicitPipeline = "ExplicitPipeline"


class LdsLayout(str, Enum):
    Canonical = "Canonical"
    PaddedRows = "PaddedRows"
