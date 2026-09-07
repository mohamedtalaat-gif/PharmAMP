"""Wrapper around OmegAMP for use as a benchmark generator.

OmegAMP's checkpoint loader (`load_model_for_inference`) rebuilds the exact
training-time architecture from its own Hydra config tree before loading
weights, and its datamodule reads the training sequence/embedding files
directly off disk. Reimplementing that loading path here would just be a
second, drifting copy of it, so this module instead drives OmegAMP's own
code in place, out of the vendored checkout at `third_party/OmegAMP`, and
only overrides the handful of config paths that need to point at files
living outside that checkout (our downloaded checkpoint and embeddings).
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")  # wandb is an unused transitive import here; avoid any login prompt

PHARMAMP_ROOT = Path(__file__).resolve().parents[2]
OMEGAMP_ROOT = PHARMAMP_ROOT / "third_party" / "OmegAMP"

DEFAULT_CHECKPOINT = PHARMAMP_ROOT / "models" / "generative_model.ckpt"
DEFAULT_EMBEDDINGS = PHARMAMP_ROOT / "data" / "generative-model-data" / "generative-model-embeddings.h5"


def _ensure_omegamp_importable() -> None:
    if not OMEGAMP_ROOT.exists():
        raise FileNotFoundError(
            f"OmegAMP checkout not found at {OMEGAMP_ROOT}. Clone it with:\n"
            f"  git clone https://github.com/szczurek-lab/OmegAMP.git {OMEGAMP_ROOT}"
        )
    if str(OMEGAMP_ROOT) not in sys.path:
        sys.path.insert(0, str(OMEGAMP_ROOT))


@contextlib.contextmanager
def _xgboost_load_model_disabled():
    """Skip loading the checkpoint's built-in XGBoost activity classifier,
    scoped to exactly the call that needs it — not a permanent process-wide
    patch, so anything else in the process that legitimately loads an
    XGBoost model later isn't silently turned into a no-op.

    `xgboost.Booster.load_model` segfaults natively whenever it runs in a
    process that already has `torch`/`pytorch_lightning` loaded (confirmed:
    the same call succeeds standalone, and only crashes after `import torch`
    — a native runtime collision between the two, not an OmegAMP or model
    file problem). `DiffusionTraining.__init__` builds this classifier
    unconditionally, but it's only read from Lightning's validation/test
    hooks (it backs `val/amp-probability`, the training-time checkpoint
    selection metric) — a bare `.sample()` call never touches it, so
    skipping the load is safe for generation-only use.
    """
    import xgboost as xgb

    original_load_model = xgb.XGBClassifier.load_model
    xgb.XGBClassifier.load_model = lambda self, model_path: None
    try:
        yield
    finally:
        xgb.XGBClassifier.load_model = original_load_model


class OmegAMPGenerator:
    """De novo AMP sequence generation via a pretrained OmegAMP checkpoint."""

    name = "OmegAMP"

    def __init__(
        self,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
        embeddings_path: Path | str = DEFAULT_EMBEDDINGS,
    ):
        for label, path in [("checkpoint", checkpoint_path), ("embeddings", embeddings_path)]:
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"OmegAMP {label} file not found at {path}. "
                    "See README.md's benchmark install step for the download command."
                )
        self.checkpoint_path = Path(checkpoint_path)
        self.embeddings_path = Path(embeddings_path)
        self._model = None

    def _load_model(self):
        _ensure_omegamp_importable()
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        from project.config import load_model_for_inference

        with initialize_config_dir(version_base=None, config_dir=str(OMEGAMP_ROOT / "config")):
            config = compose(config_name="train")

        OmegaConf.set_struct(config, False)
        config.data.original_amp_file = str(OMEGAMP_ROOT / "data" / "generative-model-data" / "generative-model-dataset.csv")
        config.data.embeddings_file = str(self.embeddings_path)
        config.task.classifier_model_path = str(OMEGAMP_ROOT / "models" / "broad-classifier.json")

        with _xgboost_load_model_disabled():
            return load_model_for_inference(config, str(self.checkpoint_path))

    def generate(self, num_samples: int = 32, batch_size: int = 32, seed: int | None = None) -> list[str]:
        """Generate `num_samples` unconditional de novo peptide sequences."""
        _ensure_omegamp_importable()
        from project.scripts.inference import generate_samples

        if self._model is None:
            self._model = self._load_model()

        sequences, _conditioning = generate_samples.main(
            generation_mode="de-novo",
            conditioning_strategy="unconditional",
            num_samples=num_samples,
            batch_size=batch_size,
            checkpoint_path=str(self.checkpoint_path),
            output_fasta=None,
            conditioning_output_path=None,
            seed=seed,
            model=self._model,
        )
        return list(sequences)
