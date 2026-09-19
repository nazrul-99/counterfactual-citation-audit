#!/usr/bin/env python3
"""
selftest.py -- self-test of the pipeline invariants.

Builds a small synthetic fixture in a temporary directory and checks every
invariant the pipeline depends on.  It needs no dataset, no GPU and no
network, and finishes in well under a minute.

Each check corresponds to a failure mode that would otherwise produce numbers
that look plausible and are wrong:

  * the splice operator touching pixels outside its mask
  * in-memory and materialised conditions diverging
  * coarse granularity double-mapping the citation distribution
  * coarse granularity inheriting a representative's effect instead of
    averaging the members
  * the bootstrap resampling frames instead of clips
  * DEV/TEST identity leakage through the pair id
  * a cache key that depends on when or where a call was made
  * the control detectors failing to reproduce their known ordering
  * the free-text lexicon matching "ear" inside "beard"

Usage:
    python scripts/selftest.py              # everything that needs no extra deps
    python scripts/selftest.py --verbose
    python scripts/selftest.py --with-mediapipe   # also exercise the real parser
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import traceback
from typing import Any, Callable, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ccaudit import common as C           # noqa: E402
from ccaudit import m3_splice as M3       # noqa: E402
from ccaudit import m4_detectors as M4    # noqa: E402
from ccaudit import m6_metrics as M6      # noqa: E402
from ccaudit import regions as R          # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(REPO_ROOT, "tests")
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS: List[Tuple[str, bool, str]] = []
VERBOSE = False


def check(name: str):
    def deco(fn: Callable[[], Any]):
        try:
            fn()
            RESULTS.append((name, True, ""))
            print(f"  [ok]   {name}")
        except AssertionError as exc:
            RESULTS.append((name, False, str(exc)))
            print(f"  [FAIL] {name}\n         {exc}")
            if VERBOSE:
                traceback.print_exc()
        except Exception as exc:
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
            print(f"  [FAIL] {name}\n         {type(exc).__name__}: {exc}")
            if VERBOSE:
                traceback.print_exc()
        return fn
    return deco


# --------------------------------------------------------------------------
# fixture: synthetic crops + label maps, written as a real index.json
# --------------------------------------------------------------------------

def make_fixture(root: str, n: int = 12, size: int = 128) -> str:
    """
    Build a parsed/ directory directly (no video, no mediapipe): a textured
    background, a face oval, and a manipulated region that is smoother than
    its surroundings, which is the artefact the control detectors key on.
    """
    import cv2

    R.set_vocab("face8")
    parsed = os.path.join(root, "parsed")
    os.makedirs(parsed, exist_ok=True)
    rng = np.random.default_rng(0)
    recs = []
    methods = ["Deepfakes", "Face2Face"]

    for i in range(n):
        method = methods[i % 2]
        pair_id = f"{i//2:03d}_{(i//2+1)%9:03d}"
        frame = i * 7

        base = rng.normal(128, 26, (size, size, 3)).clip(0, 255).astype(np.uint8)
        cx = cy = size // 2
        rx, ry = int(size * 0.30), int(size * 0.38)

        # Landmarks -> label map, using the same geometry builder as m2.
        sys.path.insert(0, TESTS_DIR)
        from fake_parser import synth_landmarks     # noqa: E402

        pts = synth_landmarks(cx, cy, rx, ry)
        lab = R.build_face8_labels(pts, (size, size))

        real = base.copy()
        cv2.ellipse(real, (cx, cy), (rx, ry), 0, 0, 360, (150, 168, 188), -1)
        real = np.clip(real.astype(np.float32)
                       + rng.normal(0, 11, real.shape), 0, 255).astype(np.uint8)

        # Manipulate one region by smoothing it (a face swap's signature).
        target = "mouth" if method == "Face2Face" else "nose"
        m = (lab == R.rid(target)).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), 1)
        fake = real.copy()
        sm = cv2.GaussianBlur(real, (0, 0), 3.0)
        # Smoother and slightly colour-shifted: a blur alone on a nearly flat
        # synthetic region can move fewer pixels than the detection threshold,
        # which would make the fixture, not the code, the thing under test.
        sm = np.clip(sm.astype(np.float32) + np.array([9.0, -7.0, 6.0]),
                     0, 255).astype(np.uint8)
        fake[m.astype(bool)] = sm[m.astype(bool)]

        d = os.path.join(parsed, method, pair_id)
        stem = os.path.join(d, f"f{frame:06d}")
        C.imwrite_jpeg(stem + "_real.jpg", real, 90)
        C.imwrite_jpeg(stem + "_fake.jpg", fake, 90)
        C.imwrite_png(stem + "_lab.png", lab)
        areas = R.region_areas(lab)
        recs.append({
            "sample_id": C.stable_id(method, pair_id, frame, "selftest"),
            "pair_id": pair_id, "method": method, "frame": frame,
            "real": stem + "_real.jpg", "fake": stem + "_fake.jpg",
            "lab": stem + "_lab.png", "parser_iou": 1.0,
            "manip_coverage": 1.0, "backend": "selftest", "vocab": "face8",
            "crop_box": [0, 0, size, size], "crop_size": size, "jpeg_q": 90,
            "areas": areas,
            "present_regions": [k for k in R.get_vocab() if areas[k] >= 64],
            "split": C.split_of(pair_id), "_target": target,
        })

    idx = os.path.join(parsed, "index.json")
    C.save_index(idx, recs, {"vocab": "face8", "crop_size": size, "jpeg_q": 90,
                             "backend": "selftest"})
    return idx


# --------------------------------------------------------------------------

def run_all(with_mediapipe: bool = False) -> int:
    tmp = tempfile.mkdtemp(prefix="ccaudit_selftest_")
    try:
        print(C.banner("counterfactual-citation-audit self-test"))
        print(f"workspace: {tmp}\n")

        # ---- regions -------------------------------------------------
        print("regions")

        @check("vocabulary sizes and ids are consistent")
        def _():
            R.set_vocab("face8")
            assert R.num_regions() == 8, R.num_regions()
            for i, n in enumerate(R.get_vocab()):
                assert R.rid(n) == i and R.region_of(i) == n
            assert len(R.letters(8)) == 8 and R.letters(8)[0] == "A"

        @check("coarse map covers every fine region exactly once")
        def _():
            cm = R.coarse_map("face8")
            assert set(cm) == set(R.get_vocab("face8")), set(cm) ^ set(R.get_vocab("face8"))
            assert R.coarse_vocab("face8") == ["eyes", "nose", "mouth", "rest"]

        @check("label maps are disjoint (a pixel has exactly one region)")
        def _():
            sys.path.insert(0, TESTS_DIR)
            from fake_parser import synth_landmarks
            lab = R.build_face8_labels(synth_landmarks(64, 64, 38, 48), (128, 128))
            vals = set(np.unique(lab).tolist())
            assert vals <= set(range(8)) | {255}, vals
            assert (lab != 255).sum() > 1000, "label map is nearly empty"

        # ---- common --------------------------------------------------
        print("\ncommon")

        @check("save_json is atomic and round-trips")
        def _():
            p = os.path.join(tmp, "a", "b.json")
            C.save_json(p, {"x": [1, 2.5, True]})
            assert C.load_json(p)["x"] == [1, 2.5, True]
            assert not [f for f in os.listdir(os.path.dirname(p))
                        if f.startswith(".tmp_")], "temp file left behind"

        @check("index paths are stored relative and resolved absolutely")
        def _():
            d = os.path.join(tmp, "relo")
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, "index.json")
            C.save_index(p, [{"sample_id": "s", "pair_id": "000_001",
                              "method": "M", "frame": 0,
                              "real": os.path.join(d, "x", "r.jpg"),
                              "fake": os.path.join(d, "x", "f.jpg"),
                              "lab": os.path.join(d, "x", "l.png")}], {})
            raw = C.load_json(p)["records"][0]
            assert not os.path.isabs(raw["real"]), raw["real"]
            moved = os.path.join(tmp, "relo_moved")
            shutil.copytree(d, moved)
            recs, _m = C.load_index(os.path.join(moved, "index.json"))
            assert recs[0]["real"].startswith(moved), recs[0]["real"]

        @check("DEV/TEST split is deterministic and cannot leak an identity")
        def _():
            assert C.split_of("033_097") == C.split_of("097_033")
            assert C.split_key("id3_id7_0004") == "id3_0004"
            assert C.split_of("033_097") == C.split_of("033_097")
            frac = np.mean([C.split_of(f"{i:03d}_{(i*7)%999:03d}") == "dev"
                            for i in range(3000)])
            assert 0.25 < frac < 0.35, f"dev fraction {frac:.3f} off target 0.30"

        @check("cluster bootstrap is wider than the naive one")
        def _():
            rng = np.random.default_rng(0)
            groups, vals = [], []
            for g in range(20):
                base = rng.normal(0.5, 1.0)
                for _ in range(8):
                    groups.append(g)
                    vals.append(base + rng.normal(0, 0.01))
            _v, lo_n, hi_n = C.bootstrap_ci(vals, None, 500, seed=0)
            _v, lo_c, hi_c = C.bootstrap_ci(vals, groups, 500, seed=0)
            assert (hi_c - lo_c) > 2 * (hi_n - lo_n), (
                f"cluster CI {hi_c-lo_c:.4f} not wider than naive "
                f"{hi_n-lo_n:.4f}: the bootstrap is resampling frames, "
                f"not clips")

        @check("Holm-Bonferroni steps down correctly")
        def _():
            assert C.holm_bonferroni([0.001, 0.04, 0.9]) == [True, False, False]
            assert C.holm_bonferroni([0.001, 0.001]) == [True, True]
            assert C.holm_bonferroni([]) == []

        @check("AUC matches known values")
        def _():
            assert abs(C.auc_score([1, 2, 3, 4], [0, 0, 1, 1]) - 1.0) < 1e-9
            assert abs(C.auc_score([4, 3, 2, 1], [0, 0, 1, 1]) - 0.0) < 1e-9
            assert abs(C.auc_score([1, 1, 1, 1], [0, 0, 1, 1]) - 0.5) < 1e-9

        @check("sharding is by content hash, stable under insertion")
        def _():
            items = [{"sample_id": f"s{i}"} for i in range(200)]
            a = {x["sample_id"] for x in C.shard(items, "1/4")}
            items2 = items + [{"sample_id": "NEW"}]
            b = {x["sample_id"] for x in C.shard(items2, "1/4")}
            assert a <= b, "adding a sample reshuffled existing shard members"
            parts = [len(C.shard(items, f"{i}/4")) for i in range(4)]
            assert sum(parts) == 200, parts

        @check("limit_samples picks the same subset regardless of input order")
        def _():
            recs = [{"sample_id": f"s{i:03d}"} for i in range(100)]
            a = {r["sample_id"] for r in C.limit_samples(recs, 20, 0)}
            b = {r["sample_id"] for r in C.limit_samples(recs[::-1], 20, 0)}
            c = {r["sample_id"] for r in C.limit_samples(recs[:60], 20, 0)}
            assert a == b, "subset depends on ordering"
            assert len(a) == 20 and len(c) == 20

        @check("JPEG round-trip is a no-op at q<=0 and lossy otherwise")
        def _():
            img = (np.random.default_rng(0).integers(0, 255, (64, 64, 3))
                   ).astype(np.uint8)
            assert np.array_equal(C.jpeg_roundtrip(img, 0), img)
            assert not np.array_equal(C.jpeg_roundtrip(img, 60), img)

        # ---- m1 ------------------------------------------------------
        print("\nm1_verify")
        from ccaudit import m1_verify as M1

        @check("alignment criterion accepts and rejects the right cases")
        def _():
            assert M1.is_aligned(0.99, 0.01)
            assert M1.is_aligned(0.85, 0.055), "loose arm should accept"
            assert not M1.is_aligned(0.85, 0.20)
            assert not M1.is_aligned(0.10, 0.30)
            assert not M1.is_aligned(float("nan"), 0.01)

        @check("border mask is a ring, not the whole frame")
        def _():
            m = M1.border_mask(100, 100, 0.15)
            assert m[0, 0] and m[99, 99] and not m[50, 50]
            assert 0.2 < m.mean() < 0.8, m.mean()

        @check("frames decode at exactly the requested indices")
        def _():
            import cv2
            p = os.path.join(tmp, "seek.mp4")
            vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 25,
                                 (64, 64))
            for i in range(40):
                vw.write(np.full((64, 64, 3), i * 6 % 255, np.uint8))
            vw.release()
            got = M1.read_frames_at(p, [0, 17, 33])
            assert sorted(got) == [0, 17, 33], sorted(got)
            for i in (0, 17, 33):
                assert abs(int(got[i].mean()) - (i * 6 % 255)) < 12, (
                    f"frame {i} decoded the wrong picture "
                    f"(mean {got[i].mean():.0f} vs {i*6%255})")

        # ---- m2 ------------------------------------------------------
        print("\nm2_parse")
        from ccaudit import m2_parse as M2

        @check("frame indices never leave the usable range")
        def _():
            for n in (5, 9, 40, 301):
                for k in (1, 2, 5):
                    idx = M2.frame_indices(n, k)
                    assert idx, (n, k)
                    assert min(idx) >= 0 and max(idx) < n, (n, k, idx)

        @check("crop_pad pads rather than clamping an out-of-frame box")
        def _():
            img = np.zeros((50, 50, 3), np.uint8)
            out = M2.crop_pad(img, (-10, -10, 40, 40))
            assert out.shape[:2] == (50, 50), out.shape

        @check("manipulation mask finds the changed region and nothing else")
        def _():
            a = np.full((80, 80, 3), 100, np.uint8)
            b = a.copy()
            b[20:40, 20:40] = 200
            m, _d = M2.manipulation_mask(a, b, sigma=1.0)
            inside = m[24:36, 24:36].mean()
            outside = m[60:78, 60:78].mean()
            assert inside > 0.8 and outside < 0.05, (inside, outside)

        # ---- m3 ------------------------------------------------------
        print("\nm3_splice")
        idx_path = make_fixture(tmp)
        recs, meta = C.load_index(idx_path)
        R.set_vocab(meta["vocab"])
        rec0 = recs[0]

        @check("operator never touches pixels outside its mask")
        def _():
            inv = M3.check_invariants(rec0, dilate=3)
            for k, v in inv["regions"].items():
                assert v["identity_mae_outside"] < 1e-9, (k, v)
                assert v["real_mae_outside"] < 1e-9, (k, v)

        @check("identity splice is a no-op inside the mask too (hard, no jpeg)")
        def _():
            inv = M3.check_invariants(rec0, dilate=3)
            for k, v in inv["regions"].items():
                assert v["identity_mae_inside"] < 1e-9, (k, v)

        @check("the splice actually changes the manipulated region")
        def _():
            inv = M3.check_invariants(rec0, dilate=3)
            tgt = rec0["_target"]
            assert inv["regions"][tgt]["real_mae_inside"] > 0.5, (
                f"splicing authentic pixels into {tgt} changed almost nothing")

        @check("in-memory conditions equal materialised ones, exactly")
        def _():
            out = os.path.join(tmp, "cond")
            entry = M3.make_conditions(rec0, out, ks=["mouth", "nose"],
                                       dilate=3, mode="poisson", jpeg_q=90)
            _r, _f, _l, per = M3.sample_conditions(
                rec0, ["mouth", "nose"], 3, "poisson", 90, with_floor=True)
            by = {p["region"]: p for p in per}
            for region, info in entry["regions"].items():
                for cond, rel in info["files"].items():
                    disk = C.imread(os.path.join(out, rel))
                    mem = by[region]["images"][cond]
                    assert np.array_equal(disk, mem), (
                        f"{region}/{cond}: materialised != in-memory")

        @check("all conditions are produced, including floor and inpaint")
        def _():
            _r, _f, _l, per = M3.sample_conditions(
                rec0, ["mouth"], 3, "poisson", 90, with_floor=True,
                inpaint_method="telea")
            got = set(per[0]["images"])
            assert got == {"real", "identity", "reverse", "floor",
                           "inpaint_fake", "inpaint_real"}, got

        @check("blend fallbacks are recorded, never silent")
        def _():
            lab = C.imread(rec0["lab"], 0)
            tiny = np.zeros_like(lab)
            tiny[0:2, 0:2] = 1
            img, used = M3.blend(np.zeros((lab.shape[0], lab.shape[1], 3), np.uint8),
                                 np.ones((lab.shape[0], lab.shape[1], 3), np.uint8),
                                 tiny, "poisson")
            assert "fallback" in used or used == "poisson", used

        @check("m3 --check reports the operator invariants on real samples")
        def _():
            import subprocess
            r = subprocess.run(
                [sys.executable, "-m", "ccaudit.m3_splice",
                 "--index", idx_path, "--check", "4"],
                capture_output=True, text=True, timeout=300,
                cwd=REPO_ROOT)
            assert r.returncode == 0, r.stdout[-600:] + r.stderr[-300:]
            assert "all invariants hold" in r.stdout, r.stdout[-400:]

        @check("dilation grows the mask monotonically")
        def _():
            lab = C.imread(rec0["lab"], 0)
            sizes = [int(M3.region_mask(lab, "mouth", d).sum())
                     for d in (0, 3, 7, 11)]
            assert sizes == sorted(sizes) and sizes[-1] > sizes[0], sizes

        # ---- m4 ------------------------------------------------------
        print("\nm4_detectors")

        @check("citation menu lists every region with fixed letters")
        def _():
            opts = M4.cite_options("face8")
            assert opts.count("\n") == 7, opts
            assert opts.startswith("A. ") and "\nH. " in opts

        @check("prompt paraphrases keep the option letters identical")
        def _():
            for v in range(5):
                assert "A." in M4.verdict_prompt(v) and "B." in M4.verdict_prompt(v)
                assert M4.cite_options("face8") in M4.cite_prompt(v, "face8")

        @check("every detector returns p in [0,1] and a normalised citation")
        def _():
            lab = C.imread(rec0["lab"], 0)
            fake = C.imread(rec0["fake"])
            for spec in ("dummy", "confabulator", "adaptive_oracle",
                         "fixed_oracle:mouth"):
                det = M4.build_detector(spec)
                if getattr(det, "needs_calibration", False):
                    det.calibrate(recs, n=8)
                p = det.predict(fake, lab)
                c = det.cite(fake, lab)
                assert 0.0 <= p <= 1.0, (spec, p)
                assert abs(sum(c.values()) - 1.0) < 1e-6, (spec, sum(c.values()))
                assert set(c) == set(R.get_vocab()), spec

        @check("controls are deterministic (same input, same output)")
        def _():
            lab = C.imread(rec0["lab"], 0)
            fake = C.imread(rec0["fake"])
            det = M4.build_detector("adaptive_oracle")
            det.calibrate(recs, n=8)
            assert det.predict(fake, lab) == det.predict(fake, lab)

        # ---- m5 ------------------------------------------------------
        print("\nm5_runner")
        from ccaudit import m5_runner as M5

        @check("cache keys depend on what was asked, not on when or where")
        def _():
            a = M5.cond_key("det", "sid", "mouth", "real", "poisson", 3, 90)
            b = M5.cond_key("det", "sid", "mouth", "real", "poisson", 3, 90)
            c = M5.cond_key("det", "sid", "mouth", "real", "poisson", 7, 90)
            d = M5.cond_key("det", "sid", "mouth", "identity", "poisson", 3, 90)
            assert a == b, "cache key is not stable"
            assert len({a, c, d}) == 3, "different requests collide"

        run_dir = os.path.join(tmp, "run")
        cfg = M5.RunConfig(tag="st", splice_floor=True, inpaint="telea",
                           vocab="face8")

        @check("runner produces one row per (sample, region) with all fields")
        def _():
            M5.run_multi(recs, ["adaptive_oracle", "fixed_oracle:mouth",
                                "confabulator", "dummy"],
                         run_dir, cfg, device="cpu", save_every=100,
                         calibrate_n=8)
            blob = C.load_json(os.path.join(run_dir, "raw_adaptive_oracle_st.json"))
            rows = blob["rows"]
            assert len(rows) == len(recs) * len(rec0["present_regions"]), len(rows)
            need = {"p_orig_fake", "p_orig_real", "p_real", "p_identity",
                    "p_reverse", "p_floor", "cite_orig", "cite_reverse",
                    "blend_used", "split", "p_inpaint_fake"}
            assert need <= set(rows[0]), need - set(rows[0])

        @check("re-running is a no-op and rebuilds rows from the cache")
        def _():
            before = C.load_json(os.path.join(run_dir,
                                              "raw_adaptive_oracle_st.json"))["rows"]
            os.remove(os.path.join(run_dir, "raw_adaptive_oracle_st.json"))
            M5.run_multi(recs, ["adaptive_oracle"], run_dir, cfg,
                         device="cpu", calibrate_n=8)
            after = C.load_json(os.path.join(run_dir,
                                             "raw_adaptive_oracle_st.json"))["rows"]
            assert len(after) == len(before), (len(after), len(before))
            ka = {(r["sample_id"], r["region"]): r["p_real"] for r in after}
            kb = {(r["sample_id"], r["region"]): r["p_real"] for r in before}
            assert ka == kb, "cached rebuild disagrees with the original run"

        # ---- m6 ------------------------------------------------------
        print("\nm6_metrics")

        @check("delta is the difference-in-differences, exactly")
        def _():
            rows, _m = M6.load_rows([os.path.join(
                run_dir, "raw_adaptive_oracle_st.json")])
            s = M6.samples_from_rows(rows)
            for sid, rec in s.items():
                for k, e in rec["regions"].items():
                    lhs = e["delta"]
                    rhs = e["p_identity"] - e["p_real"]
                    assert abs(lhs - rhs) < 1e-9, (sid, k, lhs, rhs)

        @check("coarsening preserves citation mass (no double mapping)")
        def _():
            rows, _m = M6.load_rows([os.path.join(
                run_dir, "raw_adaptive_oracle_st.json")])
            s = M6.samples_from_rows(rows)
            sc = M6._coarsen(s)
            for sid in s:
                a = sum(s[sid]["cite_orig"].values())
                b = sum(sc[sid]["cite_orig"].values())
                assert abs(a - 1.0) < 1e-6 and abs(b - 1.0) < 1e-6, (a, b)
                assert len(sc[sid]["regions"]) <= 4

        @check("coarsening averages members, never inherits a representative")
        def _():
            rows, _m = M6.load_rows([os.path.join(
                run_dir, "raw_adaptive_oracle_st.json")])
            s = M6.samples_from_rows(rows)
            sc = M6._coarsen(s)
            cmap = R.coarse_map("face8")
            sid = sorted(s)[0]
            for g, e in sc[sid]["regions"].items():
                members = [v["delta"] for k, v in s[sid]["regions"].items()
                           if cmap[k] == g]
                assert abs(e["delta"] - float(np.mean(members))) < 1e-9, (
                    f"{g}: {e['delta']} != mean{members}")
                assert e["n_fine"] == len(members)

        @check("the four controls reproduce their known ordering")
        def _():
            out = M6.analyse_all([run_dir], os.path.join(tmp, "metrics"),
                                 n_boot=300)
            by = {r["detector"]: r for r in out["results"]}
            ad = by["adaptive_oracle"]
            cf = by["confabulator"]
            du = by["dummy"]
            fx = next(v for k, v in by.items() if k.startswith("fixed_oracle"))
            assert ad["FS"] > 0.02, f"adaptive_oracle FS={ad['FS']:.4f}, must be > 0"
            assert ad["faithful_forward"], "adaptive_oracle must pass forward"
            assert abs(cf["FS"]) < abs(ad["FS"]), (
                f"confabulator FS={cf['FS']:.4f} not clearly below "
                f"adaptive {ad['FS']:.4f}")
            assert cf["AUC"] > 0.7, (
                f"confabulator AUC={cf['AUC']:.3f}: the control must detect "
                f"well while explaining badly")
            assert abs(du["FS"]) < 0.05, f"dummy FS={du['FS']:.4f}"
            assert fx["FS"] > 0.02, f"fixed_oracle FS={fx['FS']:.4f}"
            assert fx["CR_minus_prior"] <= ad["CR_minus_prior"] + 1e-9, (
                "fixed_oracle must not beat adaptive_oracle on reverse "
                "citation: a constant citation cannot beat its own prior")

        @check("floor stays low: the operator does not manufacture forgeries")
        def _():
            out = C.load_json(os.path.join(tmp, "metrics", "metrics.json"))
            for r in out["results"]:
                fp = r.get("floor_p_mean")
                if fp is None or not np.isfinite(fp):
                    continue
                assert fp < 0.95, (
                    f"{r['detector']}: floor p = {fp:.3f}. The identity splice "
                    f"on an authentic frame is being called manipulated.")

        @check("Holm across the detector zoo corrects the family")
        def _():
            out = C.load_json(os.path.join(tmp, "metrics", "metrics.json"))
            res = out["results"]
            fams = {r.get("holm_family_size") for r in res}
            assert fams == {len(res)}, (fams, len(res))
            du = next(r for r in res if r["detector"] == "dummy")
            assert du.get("FS_significant_holm_detectors") is False, (
                f"dummy survived the correction with p={du.get('FS_p')}")
            for r in res:
                if not r.get("faithful"):
                    assert not r.get("faithful_holm"), (
                        f"{r['detector']} became faithful only after correction, "
                        f"which is impossible -- Holm can only remove verdicts")

        @check("additivity residual is a paired within-cell statistic")
        def _():
            # Two dilation radii on the same samples.
            for d in (0, 11):
                cfg2 = M5.RunConfig(tag=f"dilate_{d}", dilate=d, vocab="face8")
                M5.run_multi(recs, ["adaptive_oracle"],
                             os.path.join(tmp, "abl"), cfg2, device="cpu",
                             calibrate_n=8)
            groups = M6.group_raw_files([os.path.join(tmp, "abl")])
            add = M6.additivity(groups, n_boot=200)
            assert add, "additivity produced nothing from two dilation tags"
            b = add["adaptive_oracle"]
            assert b["radii"] == [0, 11], b["radii"]
            assert b["n_paired_cells"] > 0, "no (sample, region) cell at both radii"
            # The residual must equal the difference of the two interaction terms.
            lhs = b["interaction_delta_raw"] - b["interaction_delta_seam"]
            assert abs(lhs - b["residual"]) < 0.05, (lhs, b["residual"])

        @check("the three-set Venn places every count inside the right intersection")
        def _():
            import math
            from ccaudit import m7_report as M7
            counts = {"SAO": 41, "SA.": 12, "S.O": 9, ".AO": 7,
                      "S..": 22, ".A.": 15, "..O": 11, "...": 33}
            svg = M7.svg_venn3(counts, ("a", "b", "c"), "t")
            for v in counts.values():
                if v != 33:
                    assert f">{v}<" in svg, f"count {v} not rendered"
            # Geometry: reproduce the layout and check each label's membership.
            w, h, r = 480, 380, 92.0
            cx, cy = w / 2.0, h / 2.0 + 8
            cs = [(cx, cy - 46), (cx - 52, cy + 34), (cx + 52, cy + 34)]
            pos = {"S..": (cx, cy - 86), ".A.": (cx - 88, cy + 62),
                   "..O": (cx + 88, cy + 62), "SA.": (cx - 44, cy - 6),
                   "S.O": (cx + 44, cy - 6), ".AO": (cx, cy + 58),
                   "SAO": (cx, cy + 10)}
            for key, (px, py) in pos.items():
                want = [c != "." for c in key]
                for i, ((ox, oy), inside) in enumerate(zip(cs, want)):
                    d = math.hypot(px - ox, py - oy)
                    assert (d < r) == inside, (
                        f"label {key!r} is {'outside' if inside else 'inside'} "
                        f"circle {i} but should be the opposite (d={d:.1f}, r={r})")

        @check("figures export as standalone svg files")
        def _():
            from ccaudit import m7_report as M7
            figs = {}
            metrics = C.load_json(os.path.join(tmp, "metrics", "metrics.json"))
            doc = M7.build(metrics, figures=figs)
            assert figs, "no figures were captured"
            paths = M7.export_figures(figs, os.path.join(tmp, "figs"))
            assert paths and all(p.endswith(".svg") for p in paths), paths
            head = open(paths[0]).read(200)
            assert head.startswith("<?xml") and "<svg" in head, head[:80]
            assert "src=" not in doc and "<script" not in doc, (
                "the report is not self-contained")

        # ---- m10 -----------------------------------------------------
        print("\nm10_localization")
        from ccaudit import m10_localization as M10

        @check("ground-truth region matches the manipulated fixture region")
        def _():
            ok = 0
            for rec in recs:
                r = M10.localize_sample(rec)
                assert r is not None, rec["sample_id"]
                if r["gt_region"] == rec["_target"]:
                    ok += 1
            assert ok >= len(recs) - 1, (
                f"only {ok}/{len(recs)} samples localised to the region that "
                f"was actually manipulated")

        # ---- m12 -----------------------------------------------------
        print("\nm12_text_regions")
        from ccaudit import m12_text_regions as M12

        @check("lexicon maps phrases to the right regions")
        def _():
            d, g = M12.text_to_regions("the lips look smoothed")
            assert g["named"] and max(d, key=d.get) == "mouth", d
            d, g = M12.text_to_regions("the left eye is wrong")
            assert d["left_eye"] == 1.0, d

        @check("lexicon does not match 'ear' inside 'beard'")
        def _():
            d, g = M12.text_to_regions("the beard looks odd")
            assert d.get("ears", 0) == 0.0 or not g["named"], (d, g)

        @check("unqualified 'eyes' splits mass 50/50")
        def _():
            d, _g = M12.text_to_regions("the eyes look wrong")
            assert abs(d["left_eye"] - 0.5) < 1e-9, d
            assert abs(d["right_eye"] - 0.5) < 1e-9, d

        @check("a sentence naming no region is flagged, not guessed")
        def _():
            d, g = M12.text_to_regions("something about it feels off")
            assert not g["named"], g
            assert abs(sum(d.values()) - 1.0) < 1e-9

        # ---- m13 -----------------------------------------------------
        print("\nm13_cset")
        from ccaudit import m13_cset as M13

        @check("CSET refuses to build a training set containing TEST rows")
        def _():
            try:
                M13.build_examples([run_dir], idx_path, "causal")
            except RuntimeError as exc:
                assert "DEV only" in str(exc), str(exc)
                return
            raise AssertionError("TEST rows were accepted into CSET training")

        # ---- platform constraints ------------------------------------
        print("\nplatform constraints")

        @check("the platform constraint audit reports no violations")
        def _():
            import subprocess
            script = os.path.join(SCRIPTS_DIR, "check_platform_constraints.py")
            if not os.path.exists(script):
                raise AssertionError("check_platform_constraints.py is missing")
            r = subprocess.run([sys.executable, script], capture_output=True,
                               text=True, timeout=180, cwd=REPO_ROOT)
            assert r.returncode == 0, (
                "constraint violations:\n" +
                "\n".join(l for l in r.stdout.splitlines()
                           if "[FAIL]" in l or "RESULT:" in l))

        # ---- optional: the real parser -------------------------------
        if with_mediapipe:
            print("\nmediapipe (optional)")

            @check("mediapipe Tasks API loads and finds a face")
            def _():
                model = M2.locate_landmarker("")
                parser = M2.FaceParser(model)
                import cv2
                img = C.imread(rec0["real"])
                big = cv2.resize(img, (512, 512))
                pts = parser.landmarks(big)
                parser.close()
                assert pts is None or pts.shape[0] >= 468, (
                    "landmarker returned an unexpected shape")

        # ---- summary -------------------------------------------------
        n_ok = sum(1 for _n, ok, _m in RESULTS if ok)
        n = len(RESULTS)
        print(C.banner(f"RESULT: {n_ok}/{n} CHECKS PASSED"
                       if n_ok == n else
                       f"RESULT: {n - n_ok} OF {n} CHECKS FAILED"))
        if n_ok != n:
            for name, ok, msg in RESULTS:
                if not ok:
                    print(f"  FAIL  {name}\n        {msg}")
            print("\nThe pipeline should not be run until these pass: every "
                  "check above corresponds to a way of producing numbers that "
                  "look plausible and are wrong.")
            return 1
        print("ALL CHECKS PASSED")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--with-mediapipe", action="store_true")
    a = ap.parse_args()
    VERBOSE = a.verbose
    sys.exit(run_all(a.with_mediapipe))
