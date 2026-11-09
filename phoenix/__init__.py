"""PHOENIX: the engineering layer around the ``vision`` recognition core.

PHOENIX never edits ``vision/``. It observes it (traces), measures it
(benchmarks), varies it through configuration (parameter overrides and
registered stage implementations) and proposes changes that a person approves.

Layers (docs/architecture/README.md):

    Layer 0  vision/            protected core, unchanged
    Layer 1  phoenix.stages     stage registry + configurable pipeline factory
             phoenix.params     every tuning constant, versioned (config/)
    Layer 2  phoenix.monitor    run traces (System Monitoring)
             phoenix.diagnose   weakness register (Diagnostics Engine)
             phoenix.experiments  hypothesis-driven experiments (Experiment Management)
             phoenix.bench      datasets + metrics + reports (Algorithm Evaluation)
             phoenix.adapt      change proposals from evidence (Adaptation Engine)
             phoenix.memory     ADRs, hypotheses, experiment log, STATE (Engineering Memory)
             phoenix.validate   gates, config versions, rollback (Validation and Recovery)

Command line: ``python -m phoenix --help``.
"""

__version__ = "0.1.0"
