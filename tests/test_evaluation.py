import pandas as pd
import pytest

from src.evaluation import expand_labels, score_detections


def ground_truth(rows):
    return pd.DataFrame(
        [
            {
                "start_date": pd.Timestamp(s),
                "end_date": pd.Timestamp(e),
                "metrics_affected": m,
            }
            for s, e, m in rows
        ]
    )


def detections(cells, warmup=()):
    """cells: {(date, metric): is_anomaly}. warmup cells get no baseline."""
    rows = [
        {"Date": pd.Timestamp(d), "Metric": m, "Baseline_Mean": 1.0, "Is_Anomaly": flag}
        for (d, m), flag in cells.items()
    ]
    rows += [
        {"Date": pd.Timestamp(d), "Metric": m, "Baseline_Mean": float("nan"), "Is_Anomaly": False}
        for d, m in warmup
    ]
    return pd.DataFrame(rows)


def test_expand_labels_one_row_per_day_and_metric():
    gt = ground_truth([("2025-01-01", "2025-01-01", "Revenue, Orders"),
                       ("2025-02-01", "2025-02-04", "Refunds")])
    labels = expand_labels(gt)
    assert len(labels) == 2 + 4
    assert set(labels.loc[labels["Event_ID"] == 1, "Metric"]) == {"Revenue", "Orders"}
    assert labels.loc[labels["Event_ID"] == 2, "Date"].nunique() == 4


def test_expand_labels_rejects_unknown_metric():
    with pytest.raises(ValueError, match="Nope"):
        expand_labels(ground_truth([("2025-01-01", "2025-01-01", "Revenue, Nope")]))


def test_tp_fp_fn_counts():
    labels = expand_labels(ground_truth([("2025-03-01", "2025-03-01", "Revenue, Orders")]))
    dets = detections({
        ("2025-03-01", "Revenue"): True,   # TP
        ("2025-03-01", "Orders"): False,   # FN
        ("2025-03-05", "Revenue"): True,   # FP
        ("2025-03-06", "Revenue"): False,
    })
    s, _ = score_detections(dets, labels, window_weeks=4)
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 1)
    assert s["precision"] == 0.5 and s["recall"] == 0.5 and s["f1"] == 0.5


def test_warmup_cells_are_not_scored():
    labels = expand_labels(ground_truth([("2025-01-02", "2025-01-02", "Revenue")]))
    dets = detections({("2025-03-01", "Revenue"): False}, warmup=[("2025-01-02", "Revenue")])
    s, per_event = score_detections(dets, labels, window_weeks=4)
    assert s["fn"] == 0 and s["cells_unscorable"] == 1
    assert s["events_scorable"] == 0
    assert per_event.loc[0, "Cells_Scorable"] == 0


def test_event_counts_as_detected_if_any_cell_hit():
    labels = expand_labels(ground_truth([("2025-03-01", "2025-03-03", "Revenue")]))
    dets = detections({
        ("2025-03-01", "Revenue"): False,
        ("2025-03-02", "Revenue"): True,
        ("2025-03-03", "Revenue"): False,
    })
    s, per_event = score_detections(dets, labels, window_weeks=4)
    assert s["events_detected"] == 1 and s["events_scorable"] == 1
    assert per_event.loc[0, "Cells_Detected"] == 1 and per_event.loc[0, "Cells_Labeled"] == 3


def test_echo_false_positives_are_identified():
    labels = expand_labels(ground_truth([("2025-03-01", "2025-03-01", "Revenue")]))
    dets = detections({
        ("2025-03-01", "Revenue"): True,   # TP
        ("2025-03-08", "Revenue"): True,   # FP exactly 1 week later: echo
        ("2025-03-09", "Revenue"): True,   # FP, unrelated weekday: not an echo
        ("2025-03-08", "Orders"): True,    # FP, different metric: not an echo
        ("2025-04-05", "Revenue"): True,   # FP 5 weeks later: outside window
    })
    s, _ = score_detections(dets, labels, window_weeks=4)
    assert s["fp"] == 4 and s["echo_fp"] == 1


def test_no_detections_gives_zero_scores():
    labels = expand_labels(ground_truth([("2025-03-01", "2025-03-01", "Revenue")]))
    s, _ = score_detections(detections({("2025-03-01", "Revenue"): False}), labels, 4)
    assert s["precision"] == 0.0 and s["recall"] == 0.0 and s["f1"] == 0.0


# --- event-level (incident) scoring --------------------------------------

from src.evaluation import score_incidents  # noqa: E402


def incident_table(rows):
    return pd.DataFrame([
        {"Incident_ID": i, "Start_Date": pd.Timestamp(s), "End_Date": pd.Timestamp(e),
         "Category": "conversion_drop", "Severity_Score": sev}
        for i, (s, e, sev) in enumerate(rows, start=1)
    ])


def test_incident_overlapping_event_counts_as_detection():
    gt = ground_truth([("2025-03-10", "2025-03-12", "Revenue")])
    inc = incident_table([("2025-03-11", "2025-03-11", 80.0), ("2025-06-01", "2025-06-01", 20.0)])
    s, per_event, matched = score_incidents(inc, gt)
    assert s["events_detected"] == 1 and s["event_recall"] == 1.0
    assert s["incidents_matched"] == 1 and s["incident_precision"] == 0.5
    assert list(matched) == [True, False]
    assert per_event.loc[0, "Max_Severity"] == 80.0


def test_tolerance_allows_one_day_slack():
    gt = ground_truth([("2025-03-10", "2025-03-10", "Revenue")])
    near = incident_table([("2025-03-11", "2025-03-11", 50.0)])
    far = incident_table([("2025-03-13", "2025-03-13", 50.0)])
    assert score_incidents(near, gt)[0]["events_detected"] == 1
    assert score_incidents(far, gt)[0]["events_detected"] == 0


def test_event_with_no_incident_is_missed():
    gt = ground_truth([("2025-03-10", "2025-03-10", "Revenue")])
    s, per_event, _ = score_incidents(incident_table([("2025-08-01", "2025-08-01", 10.0)]), gt)
    assert s["event_recall"] == 0.0
    assert not per_event.loc[0, "Detected"]
