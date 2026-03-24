from __future__ import annotations

import gradio as gr
from gradio.themes import Soft

from app.artifacts import VersionArtifacts, load_registry, model_card_markdown
from app.config import (
    CURATED_VERSIONS,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SEED,
    EXTRAPOLATION_BOUNDS,
)
from app.content import metrics_help_markdown, overview_markdown, tried_methods_markdown
from app.inference import metrics_table, run_comparison
from app.plots import (
    make_architecture_diagram,
    make_comparison_figure,
    make_pipeline_diagram,
)
import argparse


APP_CSS = """
.wrap {max-width: 1180px; margin: 0 auto;}
.smallnote {font-size: 0.92rem; color: #444;}
"""


def _slider_settings(
    name: str, art: VersionArtifacts, allow_extrapolation: bool
) -> tuple[float, float, float, float, bool]:
    lo, hi = art.ranges[name]
    if allow_extrapolation:
        b_lo, b_hi = EXTRAPOLATION_BOUNDS[name]
        value = max(b_lo, min(b_hi, _default_param_value(art, name)))
        step = max(1e-3, (b_hi - b_lo) / 500.0)
        return b_lo, b_hi, value, step, True

    if lo == hi:
        eps = max(1e-3, abs(lo) * 0.01)
        return lo - eps, lo + eps, lo, eps, False

    step = max(1e-3, (hi - lo) / 500.0)
    return lo, hi, _default_param_value(art, name), step, True


def _default_param_value(art: VersionArtifacts, name: str) -> float:
    lo, hi = art.ranges[name]
    if lo == hi:
        return float(lo)
    return float((lo + hi) / 2.0)


def _slider_update(name: str, art: VersionArtifacts, allow_extrapolation: bool):
    lo, hi, value, step, interactive = _slider_settings(name, art, allow_extrapolation)
    return gr.update(
        minimum=lo,
        maximum=hi,
        value=value,
        step=step,
        interactive=interactive,
    )


def _on_model_or_extrapolation_change(
    version: str,
    allow_extrapolation: bool,
    registry: dict[str, VersionArtifacts],
):
    art = registry[version]
    return (
        model_card_markdown(art),
        _slider_update("m", art, allow_extrapolation),
        _slider_update("mu", art, allow_extrapolation),
        _slider_update("K", art, allow_extrapolation),
        _slider_update("delta", art, allow_extrapolation),
        _slider_update("omega", art, allow_extrapolation),
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
        return None, [], f"### Error\n{err}"

    fig = make_comparison_figure(
        real=real,
        generated=generated,
        m=float(m),
        mu=float(mu),
        K=float(K),
        delta=float(delta),
        omega=float(omega),
    )

    rows = metrics_table(metrics)

    summary = (
        "### Run summary\n"
        f"- Samples: {int(metrics['n'])}\n"
        f"- Seed: {int(seed)}\n"
        f"- Wasserstein: {metrics['wasserstein']:.6f}\n"
        f"- KS stat / p-value: {metrics['ks_stat']:.6f} / {metrics['ks_pvalue']:.3e}\n"
        f"- CvM stat / p-value: {metrics['cvm_stat']:.6f} / {metrics['cvm_pvalue']:.3e}"
    )
    return fig, rows, summary


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
                        gr.Markdown(overview_markdown())
                        gr.Markdown(tried_methods_markdown())
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
                        allow_extra = gr.Checkbox(
                            value=False,
                            label="Allow extrapolation inputs (wider bounds)",
                        )
                        model_card = gr.Markdown(model_card_markdown(default_art))

                        m_lo, m_hi, m_val, m_step, m_inter = _slider_settings(
                            "m", default_art, False
                        )
                        mu_lo, mu_hi, mu_val, mu_step, mu_inter = _slider_settings(
                            "mu", default_art, False
                        )
                        k_lo, k_hi, k_val, k_step, k_inter = _slider_settings(
                            "K", default_art, False
                        )
                        d_lo, d_hi, d_val, d_step, d_inter = _slider_settings(
                            "delta", default_art, False
                        )
                        o_lo, o_hi, o_val, o_step, o_inter = _slider_settings(
                            "omega", default_art, False
                        )

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
                        plot_out = gr.Plot(label="QQ + Distribution comparison")
                        metrics_df = gr.Dataframe(
                            headers=["metric", "value", "meaning"],
                            datatype=["str", "number", "str"],
                            label="Metrics",
                            interactive=False,
                        )
                        summary_md = gr.Markdown(
                            "Click **Generate and Compare** to run inference."
                        )
                        gr.Markdown(metrics_help_markdown())

                on_change_outputs = [
                    model_card,
                    m_slider,
                    mu_slider,
                    k_slider,
                    delta_slider,
                    omega_slider,
                ]
                version_dd.change(
                    fn=lambda v, a: _on_model_or_extrapolation_change(v, a, registry),
                    inputs=[version_dd, allow_extra],
                    outputs=on_change_outputs,
                )
                allow_extra.change(
                    fn=lambda v, a: _on_model_or_extrapolation_change(v, a, registry),
                    inputs=[version_dd, allow_extra],
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
                    outputs=[plot_out, metrics_df, summary_md],
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
