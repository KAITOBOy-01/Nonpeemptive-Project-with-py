# streamlit_ljf_app.py
# Non-preemptive Longest Job First (LJF) Scheduler with upload, metrics, and charts
# Run: streamlit run streamlit_ljf_app.py

import io
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Non-Preemptive LJF Scheduler", page_icon="⚙️", layout="wide")

# ----------------------------- Core Types -----------------------------
@dataclass
class Proc:
    pid: str
    arrival: float
    burst: float

# ----------------------------- Helpers -----------------------------
def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower().strip() if ch.isalnum())

def validate_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and validate required columns (handles spaces/underscores/aliases)."""
    norm_map = {c: _norm(c) for c in df.columns}
    want_aliases = {
        "pid":     ["pid", "process", "processid", "process_id", "process id", "name", "proc", "p"],
        "arrival": ["arrival", "arrivaltime", "arrival_time", "arrival time", "at", "arrive", "timein", "starttime"],
        "burst":   ["burst", "bursttime", "burst_time", "burst time", "service", "duration", "runtime", "time", "bt"],
    }

    mapping = {}
    for want, aliases in want_aliases.items():
        alias_norms = {_norm(a) for a in aliases}
        for original, n in norm_map.items():
            if n in alias_norms and want not in mapping.values():
                mapping[original] = want
                break

    df = df.rename(columns=mapping)
    missing = [c for c in ["pid", "arrival", "burst"] if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing) +
            "\nAccepted header examples: PID/Process, Arrival Time/arrival_time/AT, Burst Time/burst_time/Service"
        )

    out = df[["pid", "arrival", "burst"]].copy()
    out["pid"] = out["pid"].astype(str)
    out["arrival"] = pd.to_numeric(out["arrival"], errors="coerce")
    out["burst"] = pd.to_numeric(out["burst"], errors="coerce")
    out = out.dropna()
    out = out[(out["arrival"] >= 0) & (out["burst"] > 0)]
    return out.sort_values(["arrival", "pid"]).reset_index(drop=True)

def schedule_ljf(df: pd.DataFrame, tie_breaker: str = "earliest_arrival") -> Tuple[pd.DataFrame, List[Dict]]:
    """Non-preemptive LJF with arrival times. Returns (per-process metrics, timeline segments)."""
    procs = [Proc(r.pid, float(r.arrival), float(r.burst)) for r in df.itertuples(index=False)]
    n = len(procs)
    if n == 0:
        return pd.DataFrame(), []

    current_time = 0.0
    completed = set()
    started: Dict[str, float] = {}
    finished: Dict[str, float] = {}
    burst_map = {p.pid: p.burst for p in procs}
    arrival_map = {p.pid: p.arrival for p in procs}
    timeline: List[Dict] = []

    earliest_arrival = min(p.arrival for p in procs)
    if current_time < earliest_arrival:
        timeline.append({"task": "IDLE", "start": current_time, "finish": earliest_arrival, "kind": "idle"})
        current_time = earliest_arrival

    while len(completed) < n:
        ready = [p for p in procs if p.arrival <= current_time and p.pid not in completed]
        if not ready:
            next_arrival = min(p.arrival for p in procs if p.pid not in completed)
            timeline.append({"task": "IDLE", "start": current_time, "finish": next_arrival, "kind": "idle"})
            current_time = next_arrival
            continue

        max_burst = max(p.burst for p in ready)
        candidates = [p for p in ready if p.burst == max_burst]

        if len(candidates) > 1:
            if tie_breaker == "earliest_arrival":
                min_arr = min(p.arrival for p in candidates)
                candidates = [p for p in candidates if p.arrival == min_arr]
                pick = sorted(candidates, key=lambda p: p.pid)[0]
            elif tie_breaker == "smallest_pid":
                pick = sorted(candidates, key=lambda p: p.pid)[0]
            else:  # largest_pid
                pick = sorted(candidates, key=lambda p: p.pid)[-1]
        else:
            pick = candidates[0]

        stime = current_time
        ftime = stime + pick.burst
        if pick.pid not in started:
            started[pick.pid] = stime
        finished[pick.pid] = ftime

        timeline.append({"task": pick.pid, "start": stime, "finish": ftime, "kind": "run"})
        current_time = ftime
        completed.add(pick.pid)

    # Per-process metrics
    rows = []
    for pid in [p.pid for p in procs]:
        at = arrival_map[pid]
        bt = burst_map[pid]
        stime = started[pid]
        ctime = finished[pid]
        tat = ctime - at
        wt = tat - bt
        rt = stime - at
        rows.append({
            "PID": pid,
            "Arrival": at,
            "Burst": bt,
            "Start": stime,
            "Completion": ctime,
            "Turnaround": tat,
            "Waiting": wt,
            "Response": rt,
        })
    result_df = pd.DataFrame(rows).sort_values("PID").reset_index(drop=True)

    # Aggregates
    total_burst = float(sum(p.burst for p in procs))
    makespan = max(finished.values()) - min(min(arrival_map.values()), timeline[0]["start"]) if timeline else 0.0
    cpu_util = 100.0 * (total_burst / makespan) if makespan else 100.0
    throughput = n / makespan if makespan else float(n)

    result_df.attrs["cpu_utilization"] = cpu_util
    result_df.attrs["throughput"] = throughput
    result_df.attrs["avg_turnaround"] = float(result_df["Turnaround"].mean()) if not result_df.empty else 0.0
    result_df.attrs["avg_waiting"] = float(result_df["Waiting"].mean()) if not result_df.empty else 0.0
    result_df.attrs["avg_response"] = float(result_df["Response"].mean()) if not result_df.empty else 0.0
    return result_df, timeline

def make_example(n: int = 10, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    arrival = np.sort(rng.integers(0, 10, size=n).astype(float))
    burst = rng.integers(1, 10, size=n).astype(float)
    pid = [f"P{i+1}" for i in range(n)]
    return pd.DataFrame({"pid": pid, "arrival": arrival, "burst": burst})

# ----------------------------- UI -----------------------------
st.title("⚙️ Non-Preemptive Longest Job First (LJF) Scheduler")
st.caption("Upload CSV/Excel with columns: pid, arrival, burst — or use the example.")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    tie = st.selectbox(
        "Tie-breaker (when bursts equal)",
        options=["earliest_arrival", "smallest_pid", "largest_pid"],
        index=0,
        help="How to choose among equal burst times in the ready queue.",
    )
    time_unit = st.text_input("Time unit label", value="ms")
    st.divider()
    st.subheader("Example")
    if st.button("Use example dataset"):
        st.session_state["example_df"] = make_example(10)

    # CSV template download
    template = pd.DataFrame({"pid": ["P1", "P2", "P3"], "arrival": [0, 2, 4], "burst": [5, 3, 8]})
    csv_buf = io.StringIO()
    template.to_csv(csv_buf, index=False)
    st.download_button("Download CSV template", data=csv_buf.getvalue(), file_name="ljf_template.csv", mime="text/csv")

# Load data
raw_df = None
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(("xlsx", "xls")):
            raw_df = pd.read_excel(uploaded)
        else:
            raw_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")

if raw_df is None and "example_df" in st.session_state:
    raw_df = st.session_state["example_df"].copy()

if raw_df is None:
    st.info("Upload a CSV/Excel file or click **Use example dataset** in the sidebar.")
    st.stop()

# Validate
try:
    df = validate_df(raw_df)
except Exception as e:
    st.error(str(e))
    st.stop()

# Show input
st.subheader("Input processes")
st.dataframe(df, use_container_width=True)

# Schedule
res_df, timeline = schedule_ljf(df, tie_breaker=tie)
if res_df.empty:
    st.warning("No valid rows to schedule.")
    st.stop()

# ----------------------------- KPIs (one row) -----------------------------
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("CPU Utilization", f"{res_df.attrs['cpu_utilization']:.2f}%")
with k2:
    st.metric("Throughput", f"{res_df.attrs['throughput']:.3f} / {time_unit}")
with k3:
    st.metric("Avg Turnaround", f"{res_df.attrs['avg_turnaround']:.3f} {time_unit}")
with k4:
    st.metric("Avg Waiting", f"{res_df.attrs['avg_waiting']:.3f} {time_unit}")
with k5:
    st.metric("Avg Response", f"{res_df.attrs['avg_response']:.3f} {time_unit}")

st.divider()

# ----------------------------- Results table -----------------------------
st.subheader("Per-process metrics")
st.dataframe(
    res_df.style.format({
        "Arrival": "{:.3f}",
        "Burst": "{:.3f}",
        "Start": "{:.3f}",
        "Completion": "{:.3f}",
        "Turnaround": "{:.3f}",
        "Waiting": "{:.3f}",
        "Response": "{:.3f}",
    }),
    use_container_width=True,
)

# Download results
out_csv = io.StringIO()
res_df.to_csv(out_csv, index=False)
st.download_button("Download results (CSV)", data=out_csv.getvalue(), file_name="ljf_results.csv", mime="text/csv")

# ----------------------------- Timeline (Gantt) -----------------------------
st.subheader("Timeline (Gantt)")
if timeline:
    gantt_df = pd.DataFrame(timeline)
    gantt_df["duration"] = gantt_df["finish"].astype(float) - gantt_df["start"].astype(float)

    fig = go.Figure()
    for _, row in gantt_df.iterrows():
        task = row["task"]
        start = float(row["start"])
        dur = float(row["duration"])
        kind = row.get("kind", "run")
        hover = f"{task} ({kind})<br>start: {start:.3f} {time_unit}<br>finish: {start+dur:.3f} {time_unit}"
        fig.add_trace(
            go.Bar(
                x=[dur],
                y=["CPU"],
                base=[start],
                orientation="h",
                name=str(task),
                hovertext=[hover],
                hoverinfo="text",
            )
        )
    fig.update_layout(
        barmode="stack",
        xaxis_title=f"Time ({time_unit})",
        yaxis_title="",
        legend_title_text="Task (PID / IDLE)",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------- Per-process charts -----------------------------
st.subheader("Per-process charts")
col_a, col_b, col_c = st.columns(3)
with col_a:
    bar1 = px.bar(res_df, x="PID", y="Turnaround", title=f"Turnaround Time ({time_unit})", text="Turnaround")
    bar1.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(bar1, use_container_width=True)
with col_b:
    bar2 = px.bar(res_df, x="PID", y="Waiting", title=f"Waiting Time ({time_unit})", text="Waiting")
    bar2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(bar2, use_container_width=True)
with col_c:
    bar3 = px.bar(res_df, x="PID", y="Response", title=f"Response Time ({time_unit})", text="Response")
    bar3.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(bar3, use_container_width=True)

# ----------------------------- Notes -----------------------------
st.caption(
    "Non-preemptive LJF selects the longest burst among ready processes and runs it to completion. "
    "Response Time = first start − arrival. Waiting Time = Turnaround − Burst. "
    "Throughput = number of processes / schedule makespan. "
    "CPU Utilization = (sum of bursts / makespan) × 100%."
)
