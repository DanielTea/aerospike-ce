# Renders

Views of `spec/regen.json`, committed because they are documentation rather than
build output. Everything under `out/` is generated and ignored; these are not.

Regenerate with:

```bash
cd validate
python render3d.py --spec ../spec/regen.json --gallery ../docs/renders
```

Meshing is the slow part and the views are cheap, so all eight come from three
meshing passes: the closed engine, the cutaway, and a fine wedge at the throat.

| view | shows |
|---|---|
| `01-three-quarter` | overall form: head block, cowl, spike |
| `02-side` | proportions, and the manifold ring stepped out of the cowl |
| `03-rear-quarter` | the truncated spike and the cowl lip |
| `04-head-end` | the four mounting lugs and their fixing holes |
| `05-cutaway` | annular chamber, throat, and the hollow centrebody |
| `06-into-the-section` | looking into the cut |
| `07-from-above` | the contraction and the plug surface |
| `08-throat-detail` | a 28 degree wedge at 0.11 mm: cooling channels in both walls |

The detail view is meshed undecimated. Quadric decimation reads a half
millimetre channel as noise against a smooth wall and collapses it, so the
channels vanish from exactly the picture meant to show them.
