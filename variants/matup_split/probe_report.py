#!/usr/bin/env python3
"""Every probe.json (clamp_probe.py) -> runs/eval/matup_clamp_probe/ep10/{report.json,summary.md}.

Tests the clamp-death hypotheses of docs/CLAMP_DEATH.md section 4 on the MatUp runs that
were probed: the probe stage's re-runs of shared / split (runs/matup_split/probe/<arm>/...)
and the intervention arms that probe in optim (*_zinit). Pure Python + json; runs
anywhere (scripts/matup_split.sh report runs it in the matup container).

A channel is DEAD at a probe when no texel is inside its clamp range (active = 0), so it
gets no gradient. "Trivial" deaths are left out of the counts: the bilinear init itself
sits on the lower bound (s_mean < 1e-3, e.g. metallic of a dielectric), so the channel is
dead at its correct value.

    python variants/matup_split/probe_report.py [--root runs/matup_split] [--out runs/eval/matup_clamp_probe/ep10]
"""
import argparse
import glob
import json
import os
import statistics as st

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
CHANNELS = ("R", "G", "B", "metallic", "alpha")
GROUP_OF = {"R": "a", "G": "a", "B": "a", "metallic": "rm", "alpha": "rm"}
TRIVIAL_MARGIN = 1e-3


def load(root):
    runs = []
    for p in sorted(glob.glob(os.path.join(root, "**", "probe.json"), recursive=True)):
        rel = os.path.relpath(p, root).replace(os.sep, "/").split("/")
        if rel[0] == "probe":                       # probe/<arm>/s<k>/<material>/probe.json
            source, rel = "probe", rel[1:]
        else:                                       # <arm>/s<k>/<material>/probe.json
            source = "optim"
        if len(rel) != 4:
            continue
        arm, seed, material = rel[0], rel[1], rel[2]
        probes = sorted(json.load(open(p))["probes"], key=lambda r: r["tag"])
        if probes:
            runs.append({"arm": arm, "seed": seed, "material": material, "source": source,
                         "split": "split" in arm, "probes": probes})
    return runs


def channel_story(run, ch):
    """Death / revival timeline of one channel in one run."""
    ps = run["probes"]
    act = [(p["tag"], p["channels"][ch]["active"]) for p in ps]
    first = ps[0]["channels"][ch]
    last = ps[-1]["channels"][ch]
    dead_tags = [t for t, a in act if a == 0.0]
    revived = any(a0 == 0.0 and a1 > 0.0 for (_, a0), (_, a1) in zip(act, act[1:]))
    return {"margin": first["s_mean"], "trivial": first["s_mean"] < TRIVIAL_MARGIN,
            "sigma0": first["d_std"], "mu0": first["d_mean"],
            "init_dead": act[0][1] == 0.0, "first_dead": dead_tags[0] if dead_tags else None,
            "final_dead": act[-1][1] == 0.0, "revived": revived,
            "final_side": ("below" if last["below"] > last["above"] else "above")
            if act[-1][1] == 0.0 else "", "final_active": act[-1][1],
            "timeline": act}


def frozen_groups(run):
    """Split runs: sub-networks with every channel dead at two consecutive probes, and how
    far their parameters moved meanwhile (param_step ~ 0 = frozen, CLAMP_DEATH.md H3)."""
    out = []
    ps = run["probes"]
    for g in ("a", "rm"):
        chs = [c for c in CHANNELS if GROUP_OF[c] == g]
        steps_dead, steps_alive = [], []
        for prev, cur in zip(ps, ps[1:]):
            step = (cur.get("param_step") or {}).get(g)
            if step is None:
                continue
            all_dead = all(prev["channels"][c]["active"] == 0.0 and cur["channels"][c]["active"] == 0.0
                           for c in chs)
            (steps_dead if all_dead else steps_alive).append(step)
        if steps_dead:
            out.append({"group": g, "epochs_all_dead": len(steps_dead),
                        "max_step_all_dead": max(steps_dead),
                        "median_step_alive": st.median(steps_alive) if steps_alive else None})
    return out


def fmt(v, f="%.3f"):
    return "-" if v is None else f % v


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.path.join(REPO, "runs", "matup_split"))
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "eval", "matup_clamp_probe", "ep10"))
    args = ap.parse_args()
    runs = load(args.root)
    if not runs:
        raise SystemExit("[report] no probe.json under %s - bash scripts/matup_split.sh probe" % args.root)

    rows, per_arm = [], {}
    for r in runs:
        label = r["arm"] + (" (probe re-run)" if r["source"] == "probe" else "")
        a = per_arm.setdefault(label, {"runs": 0, "runs_final_dead": 0, "init_deaths": 0,
                                       "drift_deaths": 0, "revivals": 0, "final_deaths": 0,
                                       "groups_all_dead_final": 0})
        a["runs"] += 1
        any_final = False
        for ch in CHANNELS:
            s = channel_story(r, ch)
            if s["trivial"] or s["first_dead"] is None:
                continue
            a["init_deaths"] += s["init_dead"]
            a["drift_deaths"] += (not s["init_dead"])
            a["revivals"] += s["revived"]
            a["final_deaths"] += s["final_dead"]
            any_final |= s["final_dead"]
            rows.append(dict(s, arm=label, seed=r["seed"], material=r["material"], channel=ch))
        a["runs_final_dead"] += any_final
        last = r["probes"][-1]["channels"]
        for g in ("a", "rm"):
            chs = [c for c in CHANNELS if GROUP_OF[c] == g]
            if all(last[c]["active"] == 0.0 for c in chs) and not all(
                    r["probes"][0]["channels"][c]["s_mean"] < TRIVIAL_MARGIN for c in chs):
                a["groups_all_dead_final"] += 1

    # sigma at init, albedo mean(R,G,B) and alpha, shared vs split per material (H2)
    sig = {}
    for r in runs:
        if r["arm"] not in ("matup_shared", "matup_split"):
            continue
        p0 = r["probes"][0]["channels"]
        d = sig.setdefault(r["material"], {}).setdefault(r["arm"], {"a": [], "alpha": [], "a_mu": []})
        d["a"].append(st.mean(p0[c]["d_std"] for c in ("R", "G", "B")))
        d["a_mu"].append(st.mean(abs(p0[c]["d_mean"]) for c in ("R", "G", "B")))
        d["alpha"].append(p0["alpha"]["d_std"])
    frozen = [dict(f, arm=r["arm"], seed=r["seed"], material=r["material"])
              for r in runs if r["split"] for f in frozen_groups(r)]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump({"root": args.root, "runs": len(runs), "per_arm": per_arm, "deaths": rows,
                   "sigma_init": sig, "frozen_groups": frozen}, fh, indent=1)

    L = ["# MatUp clamp probe: `matup_clamp_probe/ep10`", "",
         "docs/CLAMP_DEATH.md. Generated by variants/matup_split/probe_report.py from %d probed "
         "run(s) under runs/matup_split/. A channel is **dead** at a probe when no texel is inside "
         "its clamp range (no gradient). Tag 0 = initialisation, 1-10 = after epoch e. Deaths whose "
         "bilinear init already sits on the lower bound (mean < %g, e.g. metallic of a dielectric) "
         "are left out." % (len(runs), TRIVIAL_MARGIN), "",
         "## Per arm (H4 init vs drift, H3 revival, H5 interventions)", "",
         "| arm | runs | runs with a dead channel at the end | deaths at init (tag 0) | deaths later (drift) "
         "| revivals | dead at the end | groups fully dead at the end |", "|---|---|---|---|---|---|---|---|"]
    for label, a in per_arm.items():
        L.append("| %s | %d | %d | %d | %d | %d | %d | %d |" % (
            label, a["runs"], a["runs_final_dead"], a["init_deaths"], a["drift_deaths"],
            a["revivals"], a["final_deaths"], a["groups_all_dead_final"]))
    L += ["", "## Every death (H1, H4)", "",
          "margin = mean bilinear init s (distance to the lower bound); sigma0 / mu0 = spatial std / "
          "mean of the MLP output at init; first dead = first tag with no active texel.", "",
          "| arm | seed | material | channel | margin | sigma0 | mu0 | first dead | end | revived |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for x in sorted(rows, key=lambda x: (x["material"], x["arm"], x["seed"], x["channel"])):
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            x["arm"], x["seed"], x["material"], x["channel"], fmt(x["margin"], "%.4f"),
            fmt(x["sigma0"], "%.4f"), fmt(x["mu0"], "%+.4f"), x["first_dead"],
            ("dead (%s)" % x["final_side"]) if x["final_dead"] else "alive (%.2f)" % x["final_active"],
            "yes" if x["revived"] else ""))
    L += ["", "## Spatial spread of the MLP output at init (H2)", "",
          "sigma0 = std over texels of f(x) at tag 0, mean over seeds; albedo = mean of R, G, B. "
          "H2 predicts split << shared where the group's own input is flat.", "",
          "| material | albedo sigma0 shared | albedo sigma0 split | albedo abs(mu0) split "
          "| alpha sigma0 shared | alpha sigma0 split |", "|---|---|---|---|---|---|"]
    for m in sorted(sig):
        d = sig[m]
        g = lambda arm, k: st.mean(d[arm][k]) if arm in d and d[arm][k] else None  # noqa: E731
        L.append("| %s | %s | %s | %s | %s | %s |" % (
            m, fmt(g("matup_shared", "a"), "%.4f"), fmt(g("matup_split", "a"), "%.4f"),
            fmt(g("matup_split", "a_mu"), "%.4f"),
            fmt(g("matup_shared", "alpha"), "%.4f"), fmt(g("matup_split", "alpha"), "%.4f")))
    L += ["", "## Frozen sub-networks in split runs (H3)", "",
          "A group (a = R, G, B; rm = metallic, alpha) with every channel dead at two consecutive "
          "probes; its parameters should not move (Adam's momentum aside).", "",
          "| arm | seed | material | group | epochs all dead | max param step while all dead "
          "| median param step otherwise |", "|---|---|---|---|---|---|---|"]
    for f in frozen:
        L.append("| %s | %s | %s | %s | %d | %s | %s |" % (
            f["arm"], f["seed"], f["material"], f["group"], f["epochs_all_dead"],
            fmt(f["max_step_all_dead"], "%.2e"), fmt(f["median_step_alive"], "%.2e")))
    if not frozen:
        L.append("| (none) | | | | | | |")
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("[report] %d run(s), %d death(s) -> %s" % (len(runs), len(rows), args.out))


if __name__ == "__main__":
    main()
