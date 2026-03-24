from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

from .config import CURATED_VERSIONS, LOGS_ROOT


@dataclass(frozen=True)
class VersionArtifacts:
    version: str
    version_dir: Path
    checkpoint: Path | None
    hparams: dict[str, Any]
    eval_metadata: dict[str, Any]
    conditional_metrics: dict[str, Any]
    status: str

    @property
    def ranges(self) -> dict[str, tuple[float, float]]:
        r = self.eval_metadata.get("ranges", {})
        out: dict[str, tuple[float, float]] = {}
        for k in ["m", "mu", "K", "delta", "omega"]:
            v = r.get(k, [0.0, 0.0])
            out[k] = (float(v[0]), float(v[1]))
        return out


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_hparams_yaml(path: Path) -> dict[str, Any]:
    """Minimal parser for the hparams.yaml format produced by Lightning.

    Handles scalar key:value lines and !!python/tuple blocks used for betas.
    """
    if not path.exists():
        return {}

    lines = path.read_text(encoding="utf-8").splitlines()
    out: dict[str, Any] = {}
    i = 0

    def _coerce(s: str) -> Any:
        s = s.strip()
        if s.lower() in {"true", "false"}:
            return s.lower() == "true"
        try:
            if "." in s or "e" in s.lower():
                return float(s)
            return int(s)
        except ValueError:
            return s

    while i < len(lines):
        line = lines[i].rstrip()
        if not line or line.lstrip().startswith("#"):
            i += 1
            continue

        if ":" not in line:
            i += 1
            continue

        key, raw = line.split(":", 1)
        key = key.strip()
        raw = raw.strip()

        if raw == "!!python/tuple":
            vals: list[Any] = []
            i += 1
            while i < len(lines):
                item = lines[i].strip()
                if not item.startswith("-"):
                    break
                vals.append(_coerce(item[1:].strip()))
                i += 1
            out[key] = tuple(vals)
            continue

        out[key] = _coerce(raw)
        i += 1

    return out


def _pick_checkpoint(version_dir: Path) -> Path | None:
    ckpt_dir = version_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None

    best = ckpt_dir / "best_model.ckpt"
    if best.exists():
        return best

    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        return None
    return ckpts[-1]


def discover_versions(curated: list[str] | None = None) -> list[str]:
    requested = curated if curated is not None else CURATED_VERSIONS
    out: list[str] = []
    for v in requested:
        if (LOGS_ROOT / v).exists():
            out.append(v)
    return out


def load_version_artifacts(version: str) -> VersionArtifacts:
    version_dir = LOGS_ROOT / version
    test_results = version_dir / "test_results"

    hparams = _load_hparams_yaml(version_dir / "hparams.yaml")
    eval_metadata = _load_json(test_results / "eval_metadata.json")
    conditional_metrics = _load_json(test_results / "conditional_metrics.json")
    checkpoint = _pick_checkpoint(version_dir)

    has_eval = bool(eval_metadata) and bool(conditional_metrics)
    status = "complete" if has_eval and checkpoint else "partial"

    return VersionArtifacts(
        version=version,
        version_dir=version_dir,
        checkpoint=checkpoint,
        hparams=hparams,
        eval_metadata=eval_metadata,
        conditional_metrics=conditional_metrics,
        status=status,
    )


def load_registry(curated: list[str] | None = None) -> dict[str, VersionArtifacts]:
    versions = discover_versions(curated)
    return {v: load_version_artifacts(v) for v in versions}


def _fmt_range(name: str, r: tuple[float, float]) -> str:
    lo, hi = r
    if lo == hi:
        return f"- **{name}**: {lo:.3f}"
    return f"- **{name}**: [{lo:.3f}, {hi:.3f}]"


def model_card_markdown(art: VersionArtifacts) -> str:
    hp = art.hparams
    r = art.ranges

    ckpt = str(art.checkpoint.name) if art.checkpoint else "Not found"
    status = "Complete" if art.status == "complete" else "Partial"

    rows = [
        f"### Model {art.version} ({status})",
        f"- **Checkpoint**: {ckpt}",
        f"- **latent_dim**: {hp.get('latent_dim', 'N/A')}",
        f"- **cond_emb_dim**: {hp.get('cond_emb_dim', 'N/A')}",
        f"- **g_every_k_steps**: {hp.get('g_every_k_steps', 'N/A')}",
        f"- **lr_g / lr_d**: {hp.get('lr_g', 'N/A')} / {hp.get('lr_d', 'N/A')}",
        f"- **betas_g / betas_d**: {hp.get('betas_g', 'N/A')} / {hp.get('betas_d', 'N/A')}",
        f"- **normalize_conds**: {art.eval_metadata.get('normalize_conds', 'N/A')}",
        "#### Training ranges",
        _fmt_range("m", r["m"]),
        _fmt_range("mu", r["mu"]),
        _fmt_range("K", r["K"]),
        _fmt_range("delta", r["delta"]),
        _fmt_range("omega", r["omega"]),
    ]
    return "\n".join(rows)
