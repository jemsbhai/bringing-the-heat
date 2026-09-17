"""Meaningful policy tests: quality, slices, regression, latency, provenance."""
from copy import deepcopy
from demo import gate_checks


def fixture():
    candidate = {"n": 800, "accuracy": .88, "macro_f1": .88,
        "per_class": {name: {"recall": .85} for name in ["World", "Sports", "Business", "Sci/Tech"]}}
    evaluation = {"split_sha256": "same", "models": {"pytorch": deepcopy(candidate), "int8": deepcopy(candidate)}}
    identities = {name: {"sha256": name + "-identity"} for name in ["pytorch", "onnx", "int8"]}
    evaluation["artifacts"] = deepcopy(identities)
    benchmark = {"split_sha256": "same", "models": {"int8": {"p95_ms": 25, "weights_bytes": 70_000_000,
        "latencies_ms": [25] * 100}}, "artifacts": deepcopy(identities),
        "device": "CPU", "threads": 4, "batch_size": 1, "sequence_length": 128, "samples": 100, "warmup": 10}
    policy = {"min_test_examples": 800, "min_accuracy": .8, "min_macro_f1": .8,
        "min_class_recall": .65, "max_macro_f1_drop": .02, "max_cpu_p95_ms": 75,
        "max_weights_bytes": 104857600}
    return evaluation, benchmark, policy, identities, "same"


def test_good_candidate_passes():
    assert gate_checks(*fixture())["passed"]


def test_high_average_does_not_hide_bad_class():
    evaluation, benchmark, policy, identities, split_sha = fixture()
    evaluation["models"]["int8"]["per_class"]["Business"]["recall"] = .1
    result = gate_checks(evaluation, benchmark, policy, identities, split_sha)
    assert not result["passed"]
    assert result["checks"]["accuracy"]
    assert not result["checks"]["every_class_recall"]


def test_accuracy_alone_does_not_waive_latency():
    evaluation, benchmark, policy, identities, split_sha = fixture()
    benchmark["models"]["int8"]["p95_ms"] = 99
    assert not gate_checks(evaluation, benchmark, policy, identities, split_sha)["passed"]


def test_quantization_regression_and_split_mismatch_block_release():
    evaluation, benchmark, policy, identities, split_sha = fixture()
    evaluation["models"]["pytorch"]["macro_f1"] = .95
    benchmark["split_sha256"] = "different"
    result = gate_checks(evaluation, benchmark, policy, identities, split_sha)
    assert not result["checks"]["quantization_f1_regression"]
    assert not result["checks"]["same_heldout_split"]


def test_stale_measurements_cannot_approve_new_artifact():
    evaluation, benchmark, policy, identities, split_sha = fixture()
    identities["int8"]["sha256"] = "changed-weights-or-tokenizer"
    assert not gate_checks(evaluation, benchmark, policy, identities, split_sha)["checks"]["exact_runtime_artifacts"]


def test_protocol_and_missing_class_are_blocked():
    evaluation, benchmark, policy, identities, split_sha = fixture()
    benchmark["samples"] = 1
    del evaluation["models"]["int8"]["per_class"]["Business"]
    result = gate_checks(evaluation, benchmark, policy, identities, split_sha)
    assert not result["checks"]["benchmark_protocol"]
    assert not result["checks"]["complete_class_report"]


def test_target_thread_policy_is_frozen_and_enforced():
    evaluation, benchmark, policy, identities, split_sha = fixture()
    policy["benchmark_threads"] = 2
    assert not gate_checks(evaluation, benchmark, policy, identities, split_sha)["checks"]["benchmark_protocol"]
    benchmark["threads"] = 2
    assert gate_checks(evaluation, benchmark, policy, identities, split_sha)["passed"]
