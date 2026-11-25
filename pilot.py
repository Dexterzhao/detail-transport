"""Run from this directory: python pilot.py (Python 3.10+, NumPy, Pillow)."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text())


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
    r = np.hypot(x-0.55, y-0.5)
    h = np.exp(-48*(r-0.38)**2)
    return 0.055*np.array([0.65*(x-0.55)*h, 0.65*(y-0.5)*h, h]).T


def drivers(rest, amp):
    x = rest[:, 0]
    s = np.maximum(0.0, (x-CFG["fixed_strip_max_x"])/(1-CFG["fixed_strip_max_x"]))
    p = rest.copy()
    p[:, 0] += amp*CFG["stretch"]*s*s
    p[:, 2] += amp*CFG["bend"]*s*s
    return p


def unit(v):
    return v/np.linalg.norm(v, axis=-1, keepdims=True)


def tri_basis(vertices, faces):
    tri = vertices[faces]
    e1, e2 = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    return np.stack((e1, e2, unit(np.cross(e1, e2))), axis=-1)


def graph(rest, faces):
    edges = np.concatenate([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    i, j = np.concatenate([edges, edges[:, ::-1]]).T
    lap = np.zeros((len(rest), len(rest)))
    lap[i,j] = -1
    np.fill_diagonal(lap, -lap.sum(axis=1))
    fixed = (rest[:, 0] <= CFG["fixed_strip_max_x"]) | (rest[:, 0] >= CFG["driven_strip_min_x"])
    free = ~fixed
    return i,j,free,fixed,np.linalg.inv(lap[np.ix_(free, free)]),lap[np.ix_(free, fixed)]


def arap(rest, data, amp):
    # Positive uniform edge weights. Local/global vertex-rotation ARAP with hard handles.
    i,j,free,fixed,inv_lap,lap_fc = data
    er = rest[i] - rest[j]
    p = drivers(rest, amp)
    bc = lap_fc @ p[fixed]
    for _ in range(CFG["arap_iterations"]):
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


def extent_experiment(rest, faces, inv_b, asset):
    # All sampled boundary vertices and adjacent faces belong to the driven strip.
    # The maximum x of their displaced samples serves as a 1D orthographic contour.
    mask = np.isclose(rest[:, 0], 1)
    ids, local = np.nonzero(mask[faces])
    border = faces[ids, local]
    d = relief(rest[border], asset)
    def extent(amp, fitted_surface):
        current = drivers(rest, amp)
        x = current[border, 0]
        if fitted_surface:
            q = transport(current, faces[ids], inv_b[ids], 0.5)
            x = x + vec(q, d)[:, 0]
        return float(x.max())
    target = extent(CFG["contour_target_amplitude"], True)
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


def write_csv(name, rows):
    with (ROOT/name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    rest, faces = grid(CFG["grid_vertices_per_side"])
    graph_data = graph(rest, faces)
    inv_b = np.linalg.inv(tri_basis(rest, faces))
    center = rest[faces].mean(axis=1)
    pinned = (rest[faces][:,:,0].max(axis=1) <= CFG["fixed_strip_max_x"])
    edges, adjacent = edge_pairs(faces)
    mid = rest[edges].mean(axis=1)
    band_edges = np.isclose(mid[:,0], CFG["fixed_strip_max_x"]) & (
        pinned[adjacent[:,0]] != pinned[adjacent[:,1]])
    measures, contour, manifest = [],[],[]
    frames = [[] for _ in CFG["peak_sequence"]]
    # Repeat the solve also on the descending part of the cycle. Each target uses
    # the same deterministic initialization protocol and the unmodified rest chart.
    states = [arap(rest, graph_data, a) for a in CFG["peak_sequence"]]
    for asset in CFG["assets"]:
        ref = relief(center, asset)
        source = center+ref
        d_edge = relief(mid, asset)
        v_vertex = relief(rest, asset)
        np.savez_compressed(ROOT/(asset+".npz"), rest=rest, faces=faces,
                            uv=rest[:,:2], uv_displacement_samples=ref,
                            displacement_sample_uv=center[:,:2], rest_frames=np.eye(3),
                            protected_faces=pinned)
        manifest.append(dict(asset=asset, origin="Procedurally created for this numerical check",
                             representation="unit square; one analytical vector field; one UV chart",
                             vertices=len(rest), triangles=len(faces), protected_faces=int(pinned.sum())))
        contour.extend(extent_experiment(rest,faces,inv_b,asset))
        for policy, exponent in CFG["policies"].items():
            for index,amp in enumerate(CFG["peak_sequence"]):
                current = states[index]
                q = transport(current, faces, inv_b, exponent)
                vector = vec(q,ref)
                x = current[faces].mean(axis=1)+vector
                edge_gap = vec(q[adjacent[:,0]], d_edge)-vec(q[adjacent[:,1]], d_edge)
                protected_peak = float(np.max(np.linalg.norm(x[pinned]-source[pinned], axis=1)))
                tri = current[faces]
                signed = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])[:,2]
                failed = not (np.isfinite(x).all() and signed.min()>0
                              and protected_peak < CFG["failure_tolerance"])
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
    write_csv("measures.csv",measures)
    write_csv("contours.csv",contour)
    write_csv("asset_manifest.csv",manifest)
    animation=[]
    for tiles in frames:
        page=Image.new("RGB",(1020,870),"white")
        for k,im in enumerate(tiles):
            page.paste(im,((k%3)*340,(k//3)*290))
        animation.append(page)
    peak = int(np.argmax(CFG["peak_sequence"]))
    animation[peak].save(ROOT/"peak_comparison.png")
    animation[0].save(ROOT/"cycle_demo.gif",save_all=True,append_images=animation[1:],
                      duration=340,loop=0,optimize=False)
    peak_rows=[r for r in measures if r["step"]==peak]
    end_rows=[r for r in measures if r["step"]==len(frames)-1]
    for p in CFG["policies"]:
        group=[r for r in peak_rows if r["policy"]==p]
        print(p,[round(100*r["detail_length_change_nrmse"],6) for r in group])
    print("cycle max rms:",max(r["absolute_displaced_error_from_rest_rms"] for r in end_rows))
    print("protected max:",max(r["protected_max"] for r in measures))
    print("failures:",sum(r["geometric_check_failed"] for r in measures),"of",len(measures))
    print("contour errors:",[(r["asset"],r["fit"],r["displaced_extent_absolute_error"]) for r in contour])
    print("peak protected-edge gaps:",[(r["asset"],r["policy"],r["protected_boundary_gap_max"]) for r in peak_rows])


if __name__ == "__main__":
    main()
