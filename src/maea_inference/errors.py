"""Public exception hierarchy for the inference runtime."""


class MaeaInferenceError(Exception):
    """Base class for actionable inference errors."""


class ConfigurationError(MaeaInferenceError):
    """The supplied task configuration is invalid or unsupported."""


class CheckpointError(MaeaInferenceError):
    """The supplied checkpoint cannot be loaded safely or exactly."""


class UnsupportedModelError(ConfigurationError):
    """The requested model family is outside the standalone runtime."""


class PreprocessingError(MaeaInferenceError):
    """An input cannot be converted to the model input contract."""
