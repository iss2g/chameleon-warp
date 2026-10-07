# Tool-card art

Drop before/after images here and they appear automatically on the tool-select
cards (and can be reused in the HelpTip). Until a file exists, the card shows a
neutral "before → after" placeholder (no broken image).

Expected filenames (referenced from `src/tools.tsx`):

| File            | Tool         |
|-----------------|--------------|
| `wrap.webp`     | Wrap         |
| `region.webp`   | Wrap Region  |
| `transfer.webp` | DT Transfer  |
| `refit.webp`    | Refit        |
| `fit.webp`      | Fit          |

Recommendations:
- Format: **.webp** (small, good quality). PNG/JPG also work if you change the
  `image` path in `src/tools.tsx`.
- A single image showing **before → after** side by side reads best on a card.
- Aspect ratio ~**16:9** (e.g. 640×360 or 800×450); keep each under ~150 KB.
- Dark background matches the app theme.

Example (region): left = source with the outlined region + markers around the
ear; right = the result with the source's detailed ear conformed onto the
target, everything else untouched.
