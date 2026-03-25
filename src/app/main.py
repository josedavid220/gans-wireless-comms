from __future__ import annotations

import argparse
import numpy as np

import gradio as gr
from gradio.themes import Soft

from app.artifacts import VersionArtifacts, load_registry, model_card_markdown
from app.config import (
    CURATED_VERSIONS,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SEED,
    UI_PARAM_BOUNDS,
)
from app.content import read_markdown
from app.inference import metrics_table, run_comparison
from app.plots import (
    make_architecture_diagram,
    make_cdf_figure,
    make_density_figure,
    make_pipeline_diagram,
    make_qq_figure,
)


APP_CSS = """
.wrap {max-width: 1180px; margin: 0 auto;}
.smallnote {font-size: 0.92rem; color: #444;}

.callout-info {
    background-color: #f0f7ff;
    border-left: 5px solid #007bff;
    padding: 15px;
    margin: 10px 0;
    border-radius: 4px;
    color: #333;
}
.callout-info strong {
    color: #007bff;
    display: block;
    margin-bottom: 5px;
}
"""

LATEX_DELIMITERS = [
  {"left": "$$", "right": "$$", "display": True},
  {"left": "$", "right": "$", "display": False},
  {"left": "\\(", "right": "\\)", "display": False},
  {"left": "\\[", "right": "\\]", "display": True}
]

def _slider_settings(name: str, art: VersionArtifacts) -> tuple[float, float, float, float, bool]:
    lo_ui, hi_ui = UI_PARAM_BOUNDS[name]
    value = max(lo_ui, min(hi_ui, _default_param_value(art, name)))
    step = max(1e-3, (hi_ui - lo_ui) / 500.0)
    return lo_ui, hi_ui, value, step, True


def _default_param_value(art: VersionArtifacts, name: str) -> float:
    lo, hi = art.ranges[name]
    if lo == hi:
        return float(lo)
    return float((lo + hi) / 2.0)


def _slider_update(name: str, art: VersionArtifacts):
    lo, hi, value, step, interactive = _slider_settings(name, art)
    return gr.update(
        minimum=lo,
        maximum=hi,
        value=value,
        step=step,
        interactive=interactive,
    )


def _on_model_or_extrapolation_change(
    version: str,
    registry: dict[str, VersionArtifacts],
):
    art = registry[version]
    return (
        model_card_markdown(art),
        _slider_update("m", art),
        _slider_update("mu", art),
        _slider_update("K", art),
        _slider_update("delta", art),
        _slider_update("omega", art),
    )


def _run_callback(
    version: str,
    m: float,
    mu: float,
    K: float,
    delta: float,
    omega: float,
    n_samples: int,
    seed: int,
    registry: dict[str, VersionArtifacts],
):
    art = registry[version]
    real, generated, metrics, err = run_comparison(
        art=art,
        m=float(m),
        mu=float(mu),
        K=float(K),
        delta=float(delta),
        omega=float(omega),
        n_samples=int(n_samples),
        seed=int(seed),
    )

    if err is not None:
        return None, None, None, [], f"### Error\n{err}"

    # Metrics are computed on the full sample set in run_comparison.
    # Plotting on a stable subset reduces latency without changing metrics.
    plot_n = min(3500, real.shape[0], generated.shape[0])
    if plot_n < real.shape[0]:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(real.shape[0], size=plot_n, replace=False)
        real_plot = real[idx]
        gen_plot = generated[idx]
    else:
        real_plot = real
        gen_plot = generated

    qq_fig = make_qq_figure(
        real=real_plot,
        generated=gen_plot,
    )
    density_fig = make_density_figure(
        generated=gen_plot,
        m=float(m),
        mu=float(mu),
        K=float(K),
        delta=float(delta),
        omega=float(omega),
    )
    cdf_fig = make_cdf_figure(
        real=real_plot,
        generated=gen_plot,
    )

    rows = metrics_table(metrics)


    return qq_fig, density_fig, cdf_fig, rows


def build_app() -> gr.Blocks:
    registry = load_registry(CURATED_VERSIONS)
    if not registry:
        raise RuntimeError("No curated model versions found under logs/mftr/cgan")

    versions = list(registry.keys())
    default_version = versions[0]
    default_art = registry[default_version]

    with gr.Blocks(title="MFTR cGAN Interactive Demo") as demo:
        gr.Markdown(
            "# MFTR Conditional GAN Interactive Page\n"
            "Compact project overview + interactive model comparison against theoretical MFTR."
        )

        with gr.Tabs():
            with gr.Tab("Overview"):
                with gr.Row(elem_classes=["wrap"]):
                    with gr.Column(scale=7):
                        gr.Markdown(read_markdown("overview.md"), latex_delimiters=LATEX_DELIMITERS)
                    with gr.Column(scale=5):
                        gr.Plot(value=make_pipeline_diagram, label="Pipeline")
                        gr.Plot(
                            value=make_architecture_diagram,
                            label="Current architecture",
                        )

            with gr.Tab("Model in Action"):
                with gr.Row(elem_classes=["wrap"]):
                    with gr.Column(scale=4):
                        version_dd = gr.Dropdown(
                            choices=versions,
                            value=default_version,
                            label="Model version (curated)",
                        )
                        model_card = gr.Markdown(model_card_markdown(default_art))

                        m_lo, m_hi, m_val, m_step, m_inter = _slider_settings("m", default_art)
                        mu_lo, mu_hi, mu_val, mu_step, mu_inter = _slider_settings("mu", default_art)
                        k_lo, k_hi, k_val, k_step, k_inter = _slider_settings("K", default_art)
                        d_lo, d_hi, d_val, d_step, d_inter = _slider_settings("delta", default_art)
                        o_lo, o_hi, o_val, o_step, o_inter = _slider_settings("omega", default_art)

                        m_slider = gr.Slider(
                            label="m",
                            minimum=m_lo,
                            maximum=m_hi,
                            value=m_val,
                            step=m_step,
                            interactive=m_inter,
                        )
                        mu_slider = gr.Slider(
                            label="mu",
                            minimum=mu_lo,
                            maximum=mu_hi,
                            value=mu_val,
                            step=mu_step,
                            interactive=mu_inter,
                        )
                        k_slider = gr.Slider(
                            label="K",
                            minimum=k_lo,
                            maximum=k_hi,
                            value=k_val,
                            step=k_step,
                            interactive=k_inter,
                        )
                        delta_slider = gr.Slider(
                            label="delta",
                            minimum=d_lo,
                            maximum=d_hi,
                            value=d_val,
                            step=d_step,
                            interactive=d_inter,
                        )
                        omega_slider = gr.Slider(
                            label="omega",
                            minimum=o_lo,
                            maximum=o_hi,
                            value=o_val,
                            step=o_step,
                            interactive=o_inter,
                        )

                        n_samples = gr.Slider(
                            minimum=1000,
                            maximum=20000,
                            value=DEFAULT_NUM_SAMPLES,
                            step=500,
                            label="Number of samples",
                        )
                        seed = gr.Number(value=DEFAULT_SEED, precision=0, label="Seed")
                        run_btn = gr.Button("Generate and Compare", variant="primary")

                    with gr.Column(scale=8):
                        with gr.Row():
                            qq_plot = gr.Plot(label="QQ plot")
                            density_plot = gr.Plot(label="Histogram + KDE + MFTR PDF")
                        cdf_plot = gr.Plot(label="Empirical CDF comparison")
                        metrics_df = gr.Dataframe(
                            headers=["metric", "value", "meaning"],
                            datatype=["str", "str", "str"],
                            label="Metrics",
                            interactive=False,
                        )

                on_change_outputs = [
                    model_card,
                    m_slider,
                    mu_slider,
                    k_slider,
                    delta_slider,
                    omega_slider,
                ]
                version_dd.change(
                    fn=lambda v: _on_model_or_extrapolation_change(v, registry),
                    inputs=[version_dd],
                    outputs=on_change_outputs,
                )

                run_btn.click(
                    fn=lambda v, m, mu, k, d, o, n, s: _run_callback(
                        v, m, mu, k, d, o, n, s, registry
                    ),
                    inputs=[
                        version_dd,
                        m_slider,
                        mu_slider,
                        k_slider,
                        delta_slider,
                        omega_slider,
                        n_samples,
                        seed,
                    ],
                    outputs=[qq_plot, density_plot, cdf_plot, metrics_df],
                )

    return demo


app = build_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Launch the app publicly")
    args = parser.parse_args()

    app.launch(
        theme=Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="zinc"),
        css=APP_CSS,
        share=args.share,
    )
