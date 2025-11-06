from mlflow.tracking import MlflowClient
import mlflow
from datetime import datetime


def _trunc(s, i, cap_width):
    s = str(s)
    if i in cap_width and len(s) > cap_width[i]:
        return s[: max(0, cap_width[i] - 3)] + "..."
    return s


def _format_run_row(run):
    run_id_short = run.info.run_id[:8]
    status = run.info.status or ""
    start_time_ms = run.info.start_time
    start_time = (
        datetime.fromtimestamp(start_time_ms / 1000).isoformat(sep=" ", timespec="seconds")
        if start_time_ms
        else ""
    )
    params = ", ".join(f"{k}={v}" for k, v in run.data.params.items()) or ""
    metrics = ", ".join(f"{k}={v}" for k, v in run.data.metrics.items()) or ""
    return [run_id_short, status, start_time, params, metrics]


def _compute_widths(headers, rows, cap_width):
    widths = []
    for i, h in enumerate(headers):
        col_max = max(len(h), max((len(str(r[i])) for r in rows), default=0))
        if i in cap_width:
            col_max = min(col_max, cap_width[i])
        widths.append(col_max)
    return widths


def _print_table(headers, rows, widths, cap_width):
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(_trunc(row[i], i, cap_width).ljust(widths[i]) for i in range(len(headers))))


def _print_latest_run_metric_histories(client, latest_run):
    print("\nLatest run details (full metric histories):")
    print("Run ID:", latest_run.info.run_id)
    for metric_name in latest_run.data.metrics.keys():
        history = client.get_metric_history(latest_run.info.run_id, metric_name)
        print(f"  {metric_name}: {[ (m.value, m.timestamp) for m in history ]}")


def print_latest_run_metrics(experiment_name="Default", max_results=1000):
    # If your tracking server is local mlruns, this will work out of the box.
    # If you use a remote tracking URI set it with mlflow.set_tracking_uri("http://...") first.
    client = MlflowClient()

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"Experiment '{experiment_name}' not found.")
        return

    # get runs in the experiment, newest first
    runs = client.search_runs(
        [exp.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=max_results,
    )
    if not runs:
        print("No runs found for experiment:", experiment_name)
        return

    # prepare table rows
    rows = [_format_run_row(run) for run in runs]

    headers = ["run_id", "status", "start_time", "params", "metrics"]

    # compute column widths with caps for params/metrics
    cap_width = {3: 60, 4: 60}  # caps for params and metrics
    widths = _compute_widths(headers, rows, cap_width)

    # print table
    _print_table(headers, rows, widths, cap_width)

    # If you want full history for a metric of the latest run:
    latest_run = runs[0]
    _print_latest_run_metric_histories(client, latest_run)


if __name__ == "__main__":
    # change name if you logged to a non-default experiment
    print_latest_run_metrics(experiment_name="Default")
