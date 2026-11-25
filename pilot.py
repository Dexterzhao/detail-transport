"""Run the rest-anchored detail-transport check on procedural relief patches."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import PIL
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
ASSETS = ("beaded_band", "fluted_panel", "ridge_medallion")


def read_config(path):
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    expected = {
        "grid_vertices_per_side", "arap_iterations", "stretch", "bend", "policies",
        "fixed_strip_max_x", "driven_strip_min_x", "peak_sequence", "assets",
        "failure_tolerance", "contour_target_amplitude",
    }
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("Configuration fields must match config.json in the repository.")
    for key in ("grid_vertices_per_side", "arap_iterations"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")

    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    for key in ("stretch", "bend", "fixed_strip_max_x", "driven_strip_min_x",
                "failure_tolerance", "contour_target_amplitude"):
        if not finite(config[key]):
            raise ValueError(f"{key} must be finite.")
    if config["failure_tolerance"] <= 0 or config["stretch"] <= 0:
        raise ValueError("The tolerance and horizontal stretch must be positive.")
    if not 0 <= config["contour_target_amplitude"] <= 1:
        raise ValueError("The contour target must be in the fitted interval [0, 1].")
    if not 0 <= config["fixed_strip_max_x"] < config["driven_strip_min_x"] <= 1:
        raise ValueError("Pin and drive disjoint ordered strips within the unit patch.")
    axes = np.linspace(0, 1, config["grid_vertices_per_side"])
    pinned = axes <= config["fixed_strip_max_x"]
    driven = axes >= config["driven_strip_min_x"]
    if pinned.sum() < 2 or driven.sum() < 2 or (pinned | driven).all():
        raise ValueError("Each handle strip needs two vertex columns with at least one free column between the strips.")
    sequence = config["peak_sequence"]
    if not isinstance(sequence, list) or not sequence or not all(
        finite(amplitude) and 0 <= amplitude <= 1 for amplitude in sequence
    ):
        raise ValueError("peak_sequence must contain amplitudes in [0, 1].")
    if sequence[0] != 0 or sequence[-1] != 0 or max(sequence) == 0:
        raise ValueError("A cycle must start and end at zero and reach a positive amplitude.")
    assets = config["assets"]
    if not isinstance(assets, list) or not assets or not all(name in ASSETS for name in assets):
        raise ValueError(f"Choose fields from {', '.join(ASSETS)}.")
    if len(assets) != len(set(assets)):
        raise ValueError("Asset names must be unique.")
    policies = config["policies"]
    if not isinstance(policies, dict) or not policies or not all(
        re.fullmatch(r"[a-z][a-z0-9_]*", name) and finite(value) and 0 <= value <= 1
        for name, value in policies.items()
    ):
        raise ValueError("Each policy needs an identifier and a finite exponent in [0, 1].")
    return config


def grid(n):
    axis = np.linspace(0, 1, n)
    x, y = np.meshgrid(axis, axis)
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(n*n)))
    faces = []
    for j in range(n-1):
        for i in range(n-1):
            k = j*n+i
            faces.extend([(k, k+1, k+n+1), (k, k+n+1, k+n)])
    return points, np.array(faces)


def relief(pos, asset):
    # Local chart axes agree with world axes on the planar rest state.
    x, y = pos[:, 0], pos[:, 1]
    if asset == "beaded_band":
        h = sum(np.exp(-75*((x-u)**2 + (y-v)**2))
                for u in [0.12, 0.55, 1.0] for v in [0.25, 0.72])
        return 0.05*np.array([0.6*h, 0.3*h, h]).T
    if asset == "fluted_panel":
        wave = 0.5 + 0.5*np.cos(6*np.pi*y)
        return 0.045*np.array([wave, 0.35*np.sin(6*np.pi*y), 0.65*wave]).T
    if asset != "ridge_medallion":
        raise ValueError(f"Unknown relief field: {asset}")
    r = np.hypot(x-0.55, y-0.5)
    h = np.exp(-48*(r-0.38)**2)
    return 0.055*np.array([0.65*(x-0.55)*h, 0.65*(y-0.5)*h, h]).T


def drivers(rest, amp, config):
    x = rest[:, 0]
    edge = config["fixed_strip_max_x"]
    s = np.maximum(0.0, (x - edge) / (1 - edge))
    p = rest.copy()
    p[:, 0] += amp * config["stretch"] * s*s
    p[:, 2] += amp * config["bend"] * s*s
    return p


def unit(v):
    return v/np.linalg.norm(v, axis=-1, keepdims=True)


def tri_basis(vertices, faces):
    tri = vertices[faces]
    e1, e2 = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    return np.stack((e1, e2, unit(np.cross(e1, e2))), axis=-1)


def graph(rest, faces, config):
    edges = np.concatenate([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    i, j = np.concatenate([edges, edges[:, ::-1]]).T
    lap = np.zeros((len(rest), len(rest)))
    lap[i,j] = -1
    np.fill_diagonal(lap, -lap.sum(axis=1))
    fixed = (rest[:, 0] <= config["fixed_strip_max_x"]) | (rest[:, 0] >= config["driven_strip_min_x"])
    free = ~fixed
    return i,j,free,fixed,np.linalg.inv(lap[np.ix_(free, free)]),lap[np.ix_(free, fixed)]


def arap(rest, data, amp, config):
    # Positive uniform edge weights. Local/global vertex-rotation ARAP with hard handles.
    i,j,free,fixed,inv_lap,lap_fc = data
    er = rest[i] - rest[j]
    p = drivers(rest, amp, config)
    bc = lap_fc @ p[fixed]
    for _ in range(config["arap_iterations"]):
        cov = np.zeros((len(p), 3, 3))
        np.add.at(cov, i, (p[i]-p[j])[:, :, None]*er[:, None, :])
        u, _, vh = np.linalg.svd(cov)
        sign = np.sign(np.linalg.det(u @ vh))
        u[:, :, -1] *= sign[:, None]
        rot = u @ vh
        rhs = np.zeros_like(p)
        np.add.at(rhs, i, np.einsum("eij,ej->ei", 0.5*(rot[i]+rot[j]), er))
        p[free] = inv_lap @ (rhs[free]-bc)
    return p


def transport(current, faces, inv_rest_basis, exponent):
    j = tri_basis(current, faces) @ inv_rest_basis
    u, s, vh = np.linalg.svd(j)
    return (u*(s**exponent)[:, None, :]) @ vh


def vec(q, v):
    return np.einsum("...ij,...j->...i", q, v)


def edge_pairs(faces):
    incident = {}
    for f, tri in enumerate(faces):
        for a,b in zip(tri, np.roll(tri, -1)):
            incident.setdefault(tuple(sorted([a,b])), []).append(f)
    interior = [(e,fs) for e,fs in incident.items() if len(fs)==2]
    return np.array([e for e,_ in interior]), np.array([fs for _,fs in interior])


def rms_length(v):
    return float(np.sqrt(np.mean(np.sum(v*v, axis=-1))))


def posed_vertices(current, faces, q, d):
    v = np.zeros_like(current)
    count = np.zeros(len(current))
    for corner in faces.T:
        np.add.at(v, corner, vec(q, d[corner]))
        np.add.at(count, corner, 1)
    return current + v/count[:, None]


def visual(posed, faces, amp_text, asset, policy):
    im = Image.new("RGB", (340, 290), "#ffffff")
    draw = ImageDraw.Draw(im)
    # Fixed orthographic oblique view; relief slopes control the point shading.
    x,y,z = posed.T
    screen = np.stack((27+175*x+42*y, 232-100*y-175*z), axis=-1)
    xyz = posed[faces]
    normal = unit(np.cross(xyz[:,1]-xyz[:,0], xyz[:,2]-xyz[:,0]))
    shade = np.clip(normal @ unit(np.array([0.35, -0.55, 0.75])), 0, 1)
    for f in np.argsort((-y[faces].mean(axis=1))):
        rgb = tuple(np.round(np.array([63,85,107])+shade[f]*np.array([146,137,119])).astype(int))
        draw.polygon([tuple(p) for p in screen[faces[f]]], fill=rgb)
    draw.text((14, 13), asset.replace("_", " ").title(), fill="#172f45")
    draw.text((14, 30), f"{policy.replace('_',' ')}   |   edit amplitude {amp_text}", fill="#334a60")
    return im


def extent_experiment(rest, faces, inv_b, asset, config):
    # All sampled boundary vertices and adjacent faces belong to the driven strip.
    # The maximum x of their displaced samples serves as a 1D orthographic contour.
    mask = np.isclose(rest[:, 0], 1)
    ids, local = np.nonzero(mask[faces])
    border = faces[ids, local]
    d = relief(rest[border], asset)
    def extent(amp, fitted_surface):
        current = drivers(rest, amp, config)
        x = current[border, 0]
        if fitted_surface:
            q = transport(current, faces[ids], inv_b[ids], 0.5)
            x = x + vec(q, d)[:, 0]
        return float(x.max())
    target = extent(config["contour_target_amplitude"], True)
    rows = []
    for name, displaced in [("base", False), ("displaced", True)]:
        lo, hi = 0.0, 1.0
        if not extent(lo, displaced) <= target <= extent(hi, displaced):
            raise ValueError("Unbracketed target for " + name)
        for _ in range(45):
            mid = (lo+hi)/2
            if extent(mid, displaced) < target:
                lo = mid
            else:
                hi = mid
        amp = (lo+hi)/2
        rows.append(dict(asset=asset, fit=name, target_x=target, amplitude=amp,
                         displaced_extent_absolute_error=abs(extent(amp, True)-target)))
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config, output):
    rest, faces = grid(config["grid_vertices_per_side"])
    graph_data = graph(rest, faces, config)
    inv_b = np.linalg.inv(tri_basis(rest, faces))
    center = rest[faces].mean(axis=1)
    pinned = (rest[faces][:,:,0].max(axis=1) <= config["fixed_strip_max_x"])
    edges, adjacent = edge_pairs(faces)
    mid = rest[edges].mean(axis=1)
    band_edges = pinned[adjacent[:,0]] != pinned[adjacent[:,1]]
    measures, contour, manifest = [],[],[]
    frames = [[] for _ in config["peak_sequence"]]
    # Repeat the solve also on the descending part of the cycle. Each target uses
    # the same deterministic initialization protocol and the unmodified rest chart.
    states = [arap(rest, graph_data, a, config) for a in config["peak_sequence"]]
    for asset in config["assets"]:
        ref = relief(center, asset)
        source = center+ref
        d_edge = relief(mid, asset)
        v_vertex = relief(rest, asset)
        np.savez_compressed(output/(asset+".npz"), rest=rest, faces=faces,
                            uv=rest[:,:2], uv_displacement_samples=ref,
                            displacement_sample_uv=center[:,:2], rest_frames=np.eye(3),
                            protected_faces=pinned)
        manifest.append(dict(asset=asset, origin="Procedurally created for this numerical check",
                             representation="unit square; one analytical vector field; one UV chart",
                             vertices=len(rest), triangles=len(faces), protected_faces=int(pinned.sum())))
        contour.extend(extent_experiment(rest,faces,inv_b,asset,config))
        for policy, exponent in config["policies"].items():
            for index,amp in enumerate(config["peak_sequence"]):
                current = states[index]
                q = transport(current, faces, inv_b, exponent)
                vector = vec(q,ref)
                x = current[faces].mean(axis=1)+vector
                edge_gap = vec(q[adjacent[:,0]], d_edge)-vec(q[adjacent[:,1]], d_edge)
                protected_peak = float(np.max(np.linalg.norm(x[pinned]-source[pinned], axis=1)))
                tri = current[faces]
                signed = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])[:,2]
                failed = not (np.isfinite(x).all() and signed.min()>0
                              and protected_peak < config["failure_tolerance"])
                ratio_error = np.linalg.norm(vector[~pinned],axis=1)-np.linalg.norm(ref[~pinned],axis=1)
                row = dict(asset=asset, step=index, amplitude=amp, policy=policy,
                           detail_length_change_nrmse=float(np.sqrt(np.mean(ratio_error**2))/rms_length(ref[~pinned])),
                           protected_max=protected_peak, shared_edge_gap_rms=rms_length(edge_gap),
                           protected_boundary_gap_max=float(np.linalg.norm(edge_gap[band_edges],axis=1).max()),
                           absolute_displaced_error_from_rest_rms=rms_length(x-source),
                           geometric_check_failed=int(failed))
                measures.append(row)
                posed = posed_vertices(current,faces,q,v_vertex)
                frames[index].append(visual(posed,faces,f"{amp:.1f}",asset,policy))
    write_csv(output / "measures.csv", measures)
    write_csv(output / "contours.csv", contour)
    write_csv(output / "asset_manifest.csv", manifest)
    animation=[]
    columns = len(config["policies"])
    for tiles in frames:
        page=Image.new("RGB",(340 * columns, 290 * len(config["assets"])),"white")
        for k,im in enumerate(tiles):
            page.paste(im,((k % columns)*340,(k // columns)*290))
        animation.append(page)
    peak = int(np.argmax(config["peak_sequence"]))
    animation[peak].save(output/"peak_comparison.png")
    animation[0].save(output/"cycle_demo.gif",save_all=True,append_images=animation[1:],
                      duration=340,loop=0,optimize=False)
    peak_rows=[r for r in measures if r["step"]==peak]
    end_rows=[r for r in measures if r["step"]==len(frames)-1]
    for p in config["policies"]:
        group=[r for r in peak_rows if r["policy"]==p]
        print(p,[round(100*r["detail_length_change_nrmse"],6) for r in group])
    print("cycle max rms:",max(r["absolute_displaced_error_from_rest_rms"] for r in end_rows))
    print("protected max:",max(r["protected_max"] for r in measures))
    print("failures:",sum(r["geometric_check_failed"] for r in measures),"of",len(measures))
    print("contour errors:",[(r["asset"],r["fit"],r["displaced_extent_absolute_error"]) for r in contour])
    print("peak protected-edge gaps:",[(r["asset"],r["policy"],r["protected_boundary_gap_max"]) for r in peak_rows])


def runtime_info():
    info = {"python": platform.python_version(), "numpy": np.__version__,
            "pillow": PIL.__version__, "platform": platform.system()}
    try:
        def git(*args):
            return subprocess.check_output(["git", "-C", str(ROOT), *args], encoding="utf-8",
                                           errors="surrogateescape", stderr=subprocess.DEVNULL).strip()
        if Path(git("rev-parse", "--show-toplevel")).resolve() == ROOT:
            info["commit"] = git("rev-parse", "HEAD")
            info["worktree_dirty"] = bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        pass
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "pilot",
                        help="A new directory for this run (default: runs/pilot).")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or any(output.is_relative_to(ROOT / name) for name in ("reference", ".git")):
        parser.error("Choose a new output directory outside the reference snapshot and Git metadata.")
    try:
        config = read_config(args.config)
        environment = runtime_info()
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent, prefix=".pilot-") as temp:
            stage = Path(temp) / "result"
            stage.mkdir()
            (stage / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            (stage / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
            run(config, stage)
            stage.rename(output)
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        parser.exit(1, f"Run failed: {error}\n")
    print(f"Saved run to {output}")


if __name__ == "__main__":
    main()
