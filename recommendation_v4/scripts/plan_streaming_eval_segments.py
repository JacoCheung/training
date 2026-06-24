#!/usr/bin/env python3
"""Plan streaming-train-eval job segments with global-step eval cadence.

The current streaming loop computes EVAL_EVERY_DATA_PCT per job/segment:

    eval_interval_steps = round(pct * segment_total_anchors / global_batch)

but triggers evals on the checkpoint-restored global train step:

    global_step % eval_interval_steps == 0

This helper chooses per-segment percentages so every segment derives the same
global eval interval, while also keeping estimated job walltime under budget.
"""

import argparse
from collections import namedtuple
import csv
import json
from pathlib import Path


DEFAULT_COUNTS = (
    Path(__file__).resolve().parents[2]
    / "recommendation_v4"
    / "results"
    / "yambda_report_hub"
    / "dataset_model"
    / "counts_4k_min_history_4086.csv"
)


TsCount = namedtuple("TsCount", ["ts", "samples"])
Segment = namedtuple(
    "Segment",
    [
        "job",
        "start_ts",
        "end_ts",
        "num_train_ts",
        "samples",
        "train_batches",
        "code_interval_base_steps",
        "global_step_start",
        "global_step_end",
        "eval_steps",
        "eval_every_data_pct",
        "eval_passes",
        "estimated_seconds",
    ],
)


def parse_time_seconds(value):
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("empty time value")
    if value[-1].lower() == "s":
        return float(value[:-1])
    if value[-1].lower() == "m":
        return float(value[:-1]) * 60.0
    if value[-1].lower() == "h":
        return float(value[:-1]) * 3600.0
    parts = value.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    return float(value)


def parse_ts_range(value):
    value = value.strip()
    if ".." in value:
        start, end = value.split("..", 1)
    elif ":" in value:
        start, end = value.split(":", 1)
    elif "-" in value:
        start, end = value.split("-", 1)
    else:
        raise argparse.ArgumentTypeError(
            "TS range must look like 0..298, 0:298, or 0-298"
        )
    start_i = int(start)
    end_i = int(end)
    if end_i < start_i:
        raise argparse.ArgumentTypeError(f"bad TS range {value!r}: end < start")
    return start_i, end_i


def format_seconds(seconds):
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def read_counts(path):
    counts = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"ts", "samples"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            ts = int(row["ts"])
            counts[ts] = TsCount(ts=ts, samples=int(row["samples"]))
    return counts


def select_range(counts, ts_range):
    start, end = ts_range
    missing = [ts for ts in range(start, end + 1) if ts not in counts]
    if missing:
        shown = ", ".join(map(str, missing[:10]))
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(f"counts file is missing TS: {shown}{suffix}")
    return [counts[ts] for ts in range(start, end + 1)]


def eval_hits(global_start, global_end, interval):
    first = global_start // interval + 1
    last = global_end // interval
    return [k * interval for k in range(first, last + 1)]


def plan_segments(
    train_counts,
    eval_count,
    *,
    global_batch_size,
    job_time_seconds,
    train_sec_per_batch,
    eval_sec_per_batch,
    global_eval_percentage,
    include_segment_final_eval,
    disable_pct_when_no_scheduled_eval,
):
    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be > 0")
    if job_time_seconds <= 0:
        raise ValueError("job_time_seconds must be > 0")
    if not 0 < global_eval_percentage <= 1:
        raise ValueError("global_eval_percentage must be in (0, 1]")

    total_samples = sum(ts.samples for ts in train_counts)
    # This matches streaming_train_eval_loop's EVAL_EVERY_DATA_PCT conversion:
    # it floors after summing anchors across the requested segment.
    total_code_steps = total_samples // global_batch_size
    global_interval = max(1, round(global_eval_percentage * total_code_steps))

    eval_batches = eval_count.samples // global_batch_size
    eval_pass_seconds = eval_batches * eval_sec_per_batch
    final_eval_passes = 1 if include_segment_final_eval else 0

    segments = []
    cursor = 0
    global_step_start = 0
    while cursor < len(train_counts):
        start_idx = cursor
        seg_samples = 0
        seg_train_batches = 0
        best = None

        while cursor < len(train_counts):
            ts = train_counts[cursor]
            cand_samples = seg_samples + ts.samples
            # Training actually consumes each TS with drop_last=True, so estimate
            # train work as a per-TS floor sum.
            cand_train_batches = seg_train_batches + ts.samples // global_batch_size
            cand_global_end = global_step_start + cand_train_batches
            cand_eval_steps = eval_hits(
                global_step_start, cand_global_end, global_interval
            )
            cand_eval_passes = len(cand_eval_steps) + final_eval_passes
            cand_seconds = (
                cand_train_batches * train_sec_per_batch
                + cand_eval_passes * eval_pass_seconds
            )
            if cand_seconds <= job_time_seconds or best is None:
                seg_samples = cand_samples
                seg_train_batches = cand_train_batches
                best = (
                    cursor,
                    cand_samples,
                    cand_train_batches,
                    cand_eval_steps,
                    cand_seconds,
                )
                cursor += 1
            else:
                break

        if best is None:
            raise RuntimeError("planner made no progress")

        end_idx, seg_samples, seg_train_batches, seg_eval_steps, seg_seconds = best
        start_ts = train_counts[start_idx].ts
        end_ts = train_counts[end_idx].ts
        code_base_steps = seg_samples // global_batch_size
        if code_base_steps <= 0:
            pct = None
        elif disable_pct_when_no_scheduled_eval and len(seg_eval_steps) == 0:
            pct = 0.0
        else:
            pct = global_interval / code_base_steps

        global_step_end = global_step_start + seg_train_batches
        segments.append(
            Segment(
                job=len(segments),
                start_ts=start_ts,
                end_ts=end_ts,
                num_train_ts=end_ts - start_ts + 1,
                samples=seg_samples,
                train_batches=seg_train_batches,
                code_interval_base_steps=code_base_steps,
                global_step_start=global_step_start,
                global_step_end=global_step_end,
                eval_steps=seg_eval_steps,
                eval_every_data_pct=pct,
                eval_passes=len(seg_eval_steps) + final_eval_passes,
                estimated_seconds=seg_seconds,
            )
        )
        global_step_start = global_step_end

    summary = {
        "global_batch_size": global_batch_size,
        "train_ts_start": train_counts[0].ts,
        "train_ts_end": train_counts[-1].ts,
        "eval_ts": eval_count.ts,
        "train_samples": total_samples,
        "train_code_steps": total_code_steps,
        "train_batches_estimated": sum(ts.samples // global_batch_size for ts in train_counts),
        "global_eval_percentage": global_eval_percentage,
        "global_eval_interval_steps": global_interval,
        "eval_samples": eval_count.samples,
        "eval_batches_per_pass": eval_batches,
        "eval_seconds_per_pass": eval_pass_seconds,
        "job_time_seconds": job_time_seconds,
        "train_sec_per_batch": train_sec_per_batch,
        "eval_sec_per_batch": eval_sec_per_batch,
        "num_jobs": len(segments),
    }
    return segments, summary


def print_markdown(segments, summary):
    print(
        f"global_batch={summary['global_batch_size']}, "
        f"train_ts={summary['train_ts_start']}..{summary['train_ts_end']}, "
        f"eval_ts={summary['eval_ts']}"
    )
    print(
        f"global_eval_percentage={summary['global_eval_percentage']}, "
        f"global_eval_interval_steps={summary['global_eval_interval_steps']}, "
        f"eval_batches_per_pass={summary['eval_batches_per_pass']}, "
        f"num_jobs={summary['num_jobs']}"
    )
    print()
    print(
        "| job | START_TS | NUM_TRAIN_TS | TS | EVAL_EVERY_DATA_PCT | "
        "train batches | eval global steps | eval passes | est time |"
    )
    print("|---:|---:|---:|---|---:|---:|---|---:|---:|")
    for seg in segments:
        pct = (
            "unset"
            if seg.eval_every_data_pct is None
            else f"{seg.eval_every_data_pct:.9f}"
        )
        steps = ",".join(map(str, seg.eval_steps)) if seg.eval_steps else "-"
        print(
            f"| {seg.job} | {seg.start_ts} | {seg.num_train_ts} | "
            f"{seg.start_ts}..{seg.end_ts} | {pct} | {seg.train_batches} | "
            f"{steps} | {seg.eval_passes} | {format_seconds(seg.estimated_seconds)} |"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plan streaming-train-eval TS segments and per-job eval percentages."
    )
    parser.add_argument(
        "--counts-csv",
        type=Path,
        default=DEFAULT_COUNTS,
        help=f"CSV with columns ts,samples (default: {DEFAULT_COUNTS})",
    )
    parser.add_argument(
        "--train-ts",
        type=parse_ts_range,
        required=True,
        metavar="START..END",
        help="Inclusive global train TS range, e.g. 0..298.",
    )
    parser.add_argument(
        "--eval-ts",
        type=int,
        default=299,
        help="Fixed eval holdout TS. Default: 299.",
    )
    parser.add_argument(
        "--job-time",
        type=parse_time_seconds,
        required=True,
        help="Per-job walltime budget, e.g. 3:50:00, 230m, or seconds.",
    )
    parser.add_argument(
        "--train-sec-per-batch",
        type=float,
        required=True,
        help="Estimated train seconds per distributed batch.",
    )
    parser.add_argument(
        "--eval-sec-per-batch",
        type=float,
        default=0.35,
        help="Estimated eval seconds per distributed batch. Default: 0.35.",
    )
    parser.add_argument(
        "--global-eval-percentage",
        type=float,
        required=True,
        help="Global eval fraction, e.g. 0.05 for every 5%% of global train steps.",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=8192,
        help="Distributed/global batch size. Default: 8192.",
    )
    parser.add_argument(
        "--no-segment-final-eval",
        action="store_true",
        help="Do not budget the current code's end-of-job final eval.",
    )
    parser.add_argument(
        "--keep-pct-without-hit",
        action="store_true",
        help=(
            "Print interval/global percentage even for segments that do not cross "
            "a scheduled eval step. By default those segments print 0.0."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a markdown table.",
    )
    args = parser.parse_args()

    counts = read_counts(args.counts_csv)
    if args.eval_ts not in counts:
        raise SystemExit(f"eval TS {args.eval_ts} is missing from {args.counts_csv}")
    start_ts, end_ts = args.train_ts
    if start_ts <= args.eval_ts <= end_ts:
        raise SystemExit(
            f"eval TS {args.eval_ts} is inside train range {start_ts}..{end_ts}; "
            "use a train range that excludes the fixed eval TS."
        )

    train_counts = select_range(counts, args.train_ts)
    segments, summary = plan_segments(
        train_counts,
        counts[args.eval_ts],
        global_batch_size=args.global_batch_size,
        job_time_seconds=args.job_time,
        train_sec_per_batch=args.train_sec_per_batch,
        eval_sec_per_batch=args.eval_sec_per_batch,
        global_eval_percentage=args.global_eval_percentage,
        include_segment_final_eval=not args.no_segment_final_eval,
        disable_pct_when_no_scheduled_eval=not args.keep_pct_without_hit,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "summary": summary,
                    "segments": [segment._asdict() for segment in segments],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print_markdown(segments, summary)


if __name__ == "__main__":
    main()
