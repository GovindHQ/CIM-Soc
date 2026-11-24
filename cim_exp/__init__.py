"""
cim_exp — real-image experiment pipeline for the CIM TinyViT simulator.

    imagenette.py   official Imagenette download/extraction, verified class mapping,
                    manifests, TinyViT eval preprocessing
    checkpoints.py  hash-checked registry of pretrained / QAT-INT4 / QAT-INT8
    hardware.py     the 24-configuration hardware matrix, job identities, cache keys
    metrics.py      top-1, cross-entropy, named agreement, relative logit RMS
    engine.py       FP32 references and resumable, chunked CIM jobs
    reporting.py    the five comparison cases, summary CSVs, REPORT.md
    verify.py       pre-flight validation suite

Entry point: run_experiment.py (see EXPERIMENTS.md).
"""
