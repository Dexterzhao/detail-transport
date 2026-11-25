# A numerical check of rest-anchored geometric detail transport

Three procedural reliefs share a flat unit-square chart. The experiment asks whether the choice of a transport rule changes displacement lengths, whether pinning part of the surface ensures a continuous boundary, and how the three rules behave when the base returns to rest.

## Reproduction

Extract the archive into one directory and run `python pilot.py` in a Python environment with NumPy and Pillow. The reference run used Python 3.12.14, NumPy 2.3.5 and Pillow 13.0.0 on a Windows CPU. All asset definitions and drivers are deterministic. The program reads `config.json`, generates the three mesh-and-displacement files and rewrites the CSVs, the peak comparison and the animated sequence. Floating-point residuals can vary with the linear algebra library.

Run `python figure.py` afterward to regenerate `transport_check.png`, the proposal figure combining a UV view of each rest field with its measured length changes at maximum imposed deformation.

`asset_manifest.csv` lists the three assets. Each NPZ stores the planar rest mesh, triangle indices, vertex UVs, the common orthonormal rest frame, the locations and values of the face-centre displacement samples, and the protected-face mask. The displacement is an analytic vector field. The script includes its definition so that both faces can evaluate the same rest vector at a shared edge. The PNG shows the most deformed state. The GIF shows all 11 displayed states in order. Visualizations average the face predictions at each mesh vertex. The numerical gap measurements retain the two independent face predictions.

## Protocol

The mesh has 289 vertices and 512 consistently oriented triangles. Vertices in the left quarter are fixed. The 128 faces entirely inside this strip form the protected set. For the driven vertices with x ≥ 0.875, the horizontal and vertical (out-of-plane) offsets each equal 0.35 times the edit amplitude times the squared normalized distance from the pinned strip. Other vertices are free. All three fields use a single chart; they differ in displacement directions and spatial frequencies. The edit amplitude takes ten increments from 0 to 1 and back to 0. Each state starts from the analytic initialization and uses 35 local/global iterations of a uniform-weight vertex ARAP solve with exact handle constraints. This gives the same base to all three transport rules. A zero amplitude poses a deterministic return-to-rest control.

Each triangle defines a world-space linear map J from its two rest edge vectors and unit normal to their current counterparts. This extension preserves unit normal length. If J = U diag(s) Vᵀ, the transported rest displacement is U diag(sᵖ) Vᵀ v₀. The exponents 0, ½ and 1 give rotation-only, half stretch and the full gradient. Every update reads the original displacement, with no intermediate resampling. This setup does not include a per-detail adaptive policy, artist-selected scale targets, texture filtering, or intentional changes to the reference displacement.

For editable face centres, the length-change measure is the RMS of the differences between transported and rest vector lengths divided by the RMS rest length. For each interior edge midpoint, both incident faces predict a displaced point starting from the same rest sample. Their Euclidean distance measures a gap in this face-local representation. A separate scalar fitting check projects the displaced right-edge vertex samples onto x and uses the largest coordinate as an extent. The known reference target uses the half-stretch model at amplitude 0.7. Bisection estimates the amplitude using either the base extent or the displaced extent; both fits are scored on their displaced extent. This closed-model control has no sketch uncertainty.

## Observations

| Relief | Peak length change, rotation | Half stretch | Full gradient | Final displaced extent error after base-only fitting |
|---|---:|---:|---:|---:|
| Beaded band | <0.001% | 8.27% | 22.17% | 0.018706 |
| Fluted panel | <0.001% | 17.02% | 41.00% | 0.045500 |
| Ridge medallion | <0.001% | 1.35% | 3.95% | 0.0000425 |

The unit of the last column is the rest patch width. The matching displaced-extent fit recovered all three targets to within 2 × 10⁻¹⁵. The roughly thousandfold range of base-only errors reflects the different offsets at the observed boundary. Fitting to the same forward model that generated the target serves as a numerical sanity check.

Rotation preserves vector lengths by construction. The nonzero entries quantify how two alternative stretch rules affect these particular vector fields. A length change could be the intended response to a stretch; the experiment contains no artist preference labels. A fixed vector length also permits a pattern of displaced points to spread apart as their base positions move.

Across the nine asset–policy combinations, the largest final RMS difference between displaced face centres and their analytic rest positions was 2.30 × 10⁻¹⁶ in patch-width units. All three rest-anchored rules already close this known cycle to roundoff under the deterministic base solver. At maximum deformation the largest per-case boundary-edge gap ranged from 0.00216 to 0.02089, even though the sampled protected face centres had zero positional change. Pinning the interior and checking agreement of the two incident frames are distinct tests.

The 99 rows in `measures.csv` comprise three assets, three policies and eleven displayed states, including the initial state. All rows passed the three recorded numerical checks (finite displaced centres, positive projected base-triangle orientation, and sampled protected-centre error below 10⁻¹⁰). These checks do not measure displaced-surface self-intersection, engine validity or acceptance of the reported edge gaps. The duplicate amplitudes and shared meshes preclude treating the rows as independent observations. `contours.csv` gives the six fitted amplitudes and the resulting errors.

## What remains to test

The three synthetic fields isolate transport effects on a single planar topology. A follow-on experiment should use distinct production-like assets with genuine UV seams, overlapping protections and independently specified edit targets, including cases where a policy should stretch a groove and preserve an adjacent ornament. A history-dependent solver, an intentional edit to a detail layer, a classical detail-preserving comparator and an integrated boundary treatment may all behave differently from these controls. The present measurements support feasibility of the numerical diagnostics. They do not establish perceptual preference, lower repair cost or superiority of a proposed solver.
