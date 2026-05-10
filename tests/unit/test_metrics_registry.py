"""MetricsRegistry tests."""

from __future__ import annotations

from regime.monitoring.metrics import MetricsRegistry


def test_set_and_render_unlabelled_gauge():
    m = MetricsRegistry()
    m.set_gauge("regime_crisis_prob_21d_calibrated", 0.42, help="calibrated crisis prob")
    out = m.render()
    assert "# HELP regime_crisis_prob_21d_calibrated calibrated crisis prob" in out
    assert "# TYPE regime_crisis_prob_21d_calibrated gauge" in out
    assert "regime_crisis_prob_21d_calibrated 0.42" in out


def test_inc_counter_accumulates():
    m = MetricsRegistry()
    m.inc_counter("regime_transitions_total", 1.0)
    m.inc_counter("regime_transitions_total", 2.0)
    m.inc_counter("regime_transitions_total", 3.0)
    out = m.render()
    assert "regime_transitions_total 6.0" in out


def test_labelled_gauge_render():
    m = MetricsRegistry()
    m.set_gauge("regime_posterior_prob", 0.7, labels={"regime": "calm_bull"})
    m.set_gauge("regime_posterior_prob", 0.25, labels={"regime": "neutral"})
    m.set_gauge("regime_posterior_prob", 0.05, labels={"regime": "crisis"})
    out = m.render()
    assert 'regime_posterior_prob{regime="calm_bull"} 0.7' in out
    assert 'regime_posterior_prob{regime="neutral"} 0.25' in out
    assert 'regime_posterior_prob{regime="crisis"} 0.05' in out


def test_labelled_counter_accumulates():
    m = MetricsRegistry()
    m.inc_counter("regime_transitions_total", labels={"from": "calm_bull", "to": "neutral"})
    m.inc_counter("regime_transitions_total", labels={"from": "calm_bull", "to": "neutral"})
    out = m.render()
    assert 'regime_transitions_total{from="calm_bull",to="neutral"} 2.0' in out


def test_render_emits_help_and_type_lines_once_per_metric():
    m = MetricsRegistry()
    m.set_gauge("regime_posterior_prob", 0.5, help="state posterior", labels={"regime": "x"})
    m.set_gauge("regime_posterior_prob", 0.5, labels={"regime": "y"})
    out = m.render()
    # Exactly one HELP line and one TYPE line per metric name.
    assert out.count("# HELP regime_posterior_prob") == 1
    assert out.count("# TYPE regime_posterior_prob") == 1


def test_render_returns_valid_prometheus_format_with_trailing_newline():
    m = MetricsRegistry()
    m.set_gauge("a", 1.0)
    out = m.render()
    assert out.endswith("\n")
