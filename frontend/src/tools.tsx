/**
 * Mode-first tool metadata + the tool-selection landing screen + the per-tool
 * HelpTip popover + the step-by-step guide the sidebar wizard renders.
 *
 * All user-facing copy for "what do I do now" lives here, next to the tool it
 * belongs to, so the panels in App.tsx stay layout-only. Content is static
 * (short enough not to need a separate MD pipeline). If before/after art is
 * added later, drop webp files in public/help/ and set `image` on the tool.
 */
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { ToolKind } from './store'

/** Wizard steps. Each tool picks the subset it needs, in order. */
export type StepKey =
  | 'refs'      // upload source + target
  | 'poses'     // DT transfer only: the source poses to transfer
  | 'region'    // Wrap Region only: draw the patch outline
  | 'place'     // Refit only: put the garment on the body
  | 'refine'    // Refit only: pins / freeze paint / layers
  | 'markers'   // marker pairs
  | 'settings'  // tool parameters + the Run button
  | 'result'    // view / download

export interface StepMeta {
  key: StepKey
  /** Short label under the stepper dot. Two words max — it has ~60px. */
  label: string
  /** Imperative headline of the guide card: what to do right now. */
  title: string
  /** Numbered how-to. One action per line, in the order they're performed. */
  how: string[]
  /** One line of "why this exists", shown quieter under the how-to. */
  why?: string
  /** Skippable steps are marked in the stepper and never block progress. */
  optional?: boolean
}

export interface ToolMeta {
  key: ToolKind
  label: string
  tagline: string
  beta?: boolean
  /** Optional before/after image (e.g. '/help/wrap.webp'). */
  image?: string
  /** Plain-language "pick me if…" line on the selection screen. */
  pickIf: string
  help: {
    what: string
    when: string
    markers: string
  }
  steps: StepMeta[]
}

// --- shared step bodies ----------------------------------------------------
// Several tools need the same step with only a noun changed, so build them
// from small factories instead of copy-pasting six near-identical blocks.

const refsStep = (source: string, target: string, why: string): StepMeta => ({
  key: 'refs',
  label: 'Meshes',
  title: 'Load the two meshes',
  how: [
    `Left viewport — the SOURCE: ${source}. Drag an .obj/.fbx onto it, or use “Upload source”.`,
    `Right viewport — the TARGET: ${target}.`,
    'Orbit with the left mouse button, pan with the right, zoom with the wheel. “⊡ Frame” re-centers.',
  ],
  why,
})

const markersStep = (opts: {
  title: string
  count: string
  where: string
  optional?: boolean
  why: string
}): StepMeta => ({
  key: 'markers',
  label: 'Markers',
  title: opts.title,
  optional: opts.optional,
  how: [
    'Click a vertex on the LEFT mesh — it lights up as a pending pick.',
    'Click the SAME spot on the RIGHT mesh. The two turn into one colored pair.',
    opts.count,
    opts.where,
    'Misplaced a dot? Right-click it in the viewport to delete the pair, or press Ctrl+Z.',
    'Symmetric markers: turn it on and every pair you place also lands mirrored on the other side.',
  ],
  why: opts.why,
})

const resultStep = (what: string): StepMeta => ({
  key: 'result',
  label: 'Result',
  title: 'Check it, then download',
  how: [
    'Open “Preview results” to inspect the output next to the input.',
    'The heatmap shows how far the result sits from the target — cold is close, hot is off.',
    `Download ${what}.`,
    'Not happy? Come back to any earlier step — your markers and settings are kept.',
  ],
})

export const TOOLS: ToolMeta[] = [
  {
    key: 'wrap',
    label: 'Wrap',
    tagline: 'Conform your source mesh onto a target shape, keeping source topology.',
    image: '/help/wrap.webp',
    pickIf: 'You want your own mesh to take another shape and stay riggable.',
    help: {
      what: 'Deforms the whole source so it takes on the target’s shape while keeping the source’s vertex count and topology. Output is one OBJ you can import as a customization shape key.',
      when: 'Making a character variant (wider cheeks, different head) from a base mesh, or matching one head scan to your base topology.',
      markers: '10–40 well-spread pairs on stable features (eye corners, nose tip, ear roots, jaw). Symmetric mode halves the clicking.',
    },
    steps: [
      refsStep(
        'your mesh, the one whose topology and vertex order you keep',
        'the shape you want to copy',
        'The result has the source’s topology and the target’s shape — so it drops straight into your rig as a shape key.',
      ),
      markersStep({
        title: 'Pair up landmarks on both meshes',
        count: 'Aim for 10–40 pairs. Fewer works when the meshes are already similar; more when they are not.',
        where: 'Put them on features you can find on BOTH meshes — eye corners, nose tip, ear roots, jaw line, chin — and spread them over the whole mesh. A cluster in one area leaves the rest unguided.',
        why: 'Markers are the only thing that tells the solver which part of the source belongs to which part of the target.',
      }),
      {
        key: 'settings',
        label: 'Wrap',
        title: 'Set the strength and wrap',
        how: [
          'Start with the defaults — they are tuned for a typical head/body pair.',
          'Keep “Align target to source” on unless the two meshes are already in the same scale and position.',
          '“Smooth result” removes the fine wobble the surface-matching step leaves behind. Raise it if the result looks noisy, lower it if detail is getting lost.',
          'Open Advanced only if the result missed: Loose for meshes that differ a lot or have holes, Tight for clean, similar pairs.',
          'Press Wrap. You can keep orbiting while it runs.',
        ],
      },
      resultStep('the wrapped mesh as .obj'),
    ],
  },
  {
    key: 'region',
    label: 'Wrap Region',
    tagline: 'Wrap only a patch you outline; freeze everything outside it.',
    image: '/help/region.webp',
    pickIf: 'Only one part of the mesh should change — the rest must not move.',
    help: {
      what: 'Like Wrap, but only inside a boundary loop you draw on the source. Everything outside stays exactly put, with a feathered seam whose sharpness you can tune.',
      when: 'You only want to change part of the mesh — e.g. borrow a detailed ear onto a plainer head, or reshape a nose or cheek — without disturbing the rest.',
      markers: 'A few pairs inside the patch are enough (none if source and target share topology). Draw the outline with the Outline tool, then close the loop.',
    },
    steps: [
      refsStep(
        'the mesh you are editing',
        'the shape the patch should take',
        'Only the patch you outline moves; everything outside keeps its exact position.',
      ),
      {
        key: 'region',
        label: 'Outline',
        title: 'Draw the patch you want to change',
        how: [
          'Press “✎ Outline”, then click points around the area on the LEFT mesh. The line follows the surface between your clicks.',
          'Place a point every time the boundary turns. 6–12 points is usually plenty.',
          'Close the ring by clicking the first (magenta) point again, or press “Close loop”.',
          'The smaller enclosed side becomes editable. If the wrong side lit up, press “⇄ Invert”.',
          'Press “Preview region” to color it: green = will be edited, grey = frozen, in-between = the feathered seam.',
        ],
        why: 'The loop splits the mesh into edited / frozen; the feather blends the two so the seam does not show.',
      },
      markersStep({
        title: 'Guide the patch (optional)',
        optional: true,
        count: 'A few pairs INSIDE the patch is enough. None at all if source and target already share topology.',
        where: 'Put them on features inside the outlined area. Markers outside the patch have nothing to move.',
        why: 'The frozen boundary already registers the meshes, so markers here only correct where the patch lands.',
      }),
      {
        key: 'settings',
        label: 'Wrap',
        title: 'Tune the seam and wrap',
        how: [
          '“Feather” is how wide the blend band is, as a share of the mesh size. Wider = softer transition, but the patch has less freedom near the edge.',
          '“Seam sharpness” concentrates that blend near the boundary: 1 is a gentle ramp, 5 clamps the seam hard and frees the interior sooner.',
          'Press “Wrap region”.',
        ],
      },
      resultStep('the full mesh with the patch replaced, as .obj'),
    ],
  },
  {
    key: 'transfer',
    label: 'DT Transfer',
    tagline: 'Transfer source pose/blendshape deformations onto the target.',
    image: '/help/transfer.webp',
    pickIf: 'You already animated one character and want the same expressions on another.',
    help: {
      what: 'Classic deformation transfer: takes your source poses (or FBX blendshapes) and reproduces the same deformations on the target mesh, which keeps the target’s topology.',
      when: 'You rigged/blendshaped one character and want the same expressions on another mesh.',
      markers: '≥3 pairs; more for dissimilar shapes. If source and target share identical topology, no markers are needed at all.',
    },
    steps: [
      refsStep(
        'the character your poses belong to, in its NEUTRAL pose',
        'the character that should learn those poses',
        'Both references must be neutral — the difference between neutral and a pose is exactly what gets transferred.',
      ),
      {
        key: 'poses',
        label: 'Poses',
        title: 'Add the deformations to transfer',
        how: [
          'Drop one or many .obj files of the SOURCE in different poses onto the panel, or pick them with the button.',
          'Every pose must have the same vertex and face count as the source reference — it is the same mesh, just moved.',
          'Uploaded the source as an .fbx? Its shape keys were already extracted as poses; you can add more here.',
          'Name the files the way you want the results named — the name carries through to the output.',
        ],
        why: 'Each pose is one expression/blendshape. The transfer reproduces each one on the target.',
      },
      markersStep({
        title: 'Pair up landmarks on both characters',
        count: 'At least 3 pairs. Use 15–40 when the two characters differ in proportions.',
        where: 'Put them where deformation matters — mouth corners, eyelids, brows, jaw, nostrils — plus a few on stable parts (ear roots, skull) to anchor the rest.',
        why: 'The pairs build the triangle-to-triangle mapping the transfer runs on. Identical topology skips this step entirely.',
      }),
      {
        key: 'settings',
        label: 'Run',
        title: 'Run the transfer',
        how: [
          'More Iterations = closer fit, slower run. 8 is a good default.',
          'Smoothness resists sharp local distortion; raise it if the result shows spikes.',
          'Press “Run (DT poses)”. Every pose is transferred in one job.',
        ],
      },
      resultStep('single poses as .obj, everything as .zip, or an .fbx with the shape keys built in'),
    ],
  },
  {
    key: 'refit',
    label: 'Refit',
    beta: true,
    tagline: 'Body-aware garment fit: mark the interface, keep the shape, resolve the body.',
    image: '/help/refit.webp',
    pickIf: 'You are dressing a character — cloth, armor or a seam-attached piece.',
    help: {
      what: 'Fits clothing / armor / accessories (source) onto a BODY (target) with full body awareness: your marker pairs anchor the interface, and around them come a contact field, layer classification (which walls are driven vs. which merely follow), an ARAP bind that keeps the garment’s own shape and relief, and collision (inflate the garment, or hide the body underneath). A preset picks the behavior.',
      when: 'Dressing a character — loose cloth that drapes, rigid armor over the body, or an accessory grafted at its seam. It reasons about the body surface. If you just need to pin an accessory onto a few marker points with NO body interaction, use Fit instead.',
      markers: 'The main mechanism: pairs along the interface (a shirt’s collar and hem, cuffs, the seam an accessory attaches at) decide where the garment lands. Placement matters too — pressed against the body reads as tight, left hanging as loose — and the contact/layer machinery fills in the rest. Poke is the alternative to placing pairs by hand: once the garment already sits close to the body, a needle through it derives the correspondence, gripping where the inner layer meets the body and marking any outer wall it pierces as “preserve”. Turn on “Layers: auto” for the score-based driven/follower classifier (robust to flipped game-asset normals); Preview grip then colors followers and “→ frozen paint” prefills the brush with them.',
    },
    steps: [
      refsStep(
        'the garment, armor or accessory you want to put on',
        'the BODY it goes onto',
        'Refit reads the body surface — grip, contact and collision all come from it, so the target must be the body, not another garment.',
      ),
      {
        key: 'place',
        label: 'Place',
        title: 'Put the garment where it belongs',
        how: [
          'The body appears ghosted in the left viewport so you can see through it.',
          'Press ⤧ Move / ⟳ Rotate / ⤢ Scale, then drag the gizmo handles. The camera stays put while you drag a handle.',
          'Aim for “roughly right”: the garment overlapping the body where it should touch, hanging free where it should drape.',
          'How close you leave it MATTERS — pressed against the body reads as tight, left standing off reads as loose.',
          '“↺ Reset” snaps it back to where the file put it. “Snap placement to body” lets the solver nudge your placement onto the contact standoff.',
        ],
        why: 'Placement sets the starting pose and how tight the garment reads. The next step is where you pin down what has to land where.',
      },
      {
        key: 'refine',
        label: 'Refine',
        title: 'Tie the garment to the body',
        how: [
          'Marker pairs are the main tool: click a vertex on the garment (LEFT), then the spot on the body (RIGHT) it must sit at. They turn one color in both viewports and the list below holds them all.',
          'Work along the interface — collar, hem, cuffs, the seam an accessory attaches at. Those are the places whose position you actually care about; 6–20 well-spread pairs beat a dense cluster.',
          '📌 Poke is the alternative when the garment already sits close to the body: a needle through it derives the correspondence for you — it grips where the inner layer meets the body, and any outer wall it pierces is marked “preserve” so a double-wall collar keeps its standoff.',
          '◉ Preview grip is a dry run — it colors the garment without solving: red = gripped onto the body, blue = left free. Check it before and after you change anything.',
          '❄ Freeze paint: brush over anything that must not deform at all — a buckle, a rigid collar. Left-drag paints, right-drag erases, middle-mouse orbits. “Layers: auto” + “→ frozen paint” prefills the brush with the follower walls it detected.',
          'Any change clears the grip preview — run it again to see the effect.',
        ],
        why: 'This is where you say what has to land where. The contact field, layer split and ARAP bind fill in everything you do not pin down.',
      },
      {
        key: 'settings',
        label: 'Refit',
        title: 'Pick the material behavior and fit',
        how: [
          'Accessory — grips at its seam, the bulk keeps its shape (beard, fur trim, pouch).',
          'Cloth — drapes loosely; the body is pushed out where it pokes through.',
          'Armor — very stiff, keeps its form; the body underneath is hidden instead of inflating the shell.',
          'Skin-tight — conforms everywhere, like a second skin.',
          'Press Refit. Advanced (contact distances, thickness) is only worth opening after a preset came out wrong.',
        ],
      },
      resultStep('the fitted garment as .obj'),
    ],
  },
  {
    key: 'fit',
    label: 'Fit',
    beta: true,
    tagline: 'Marker-driven attach: pin an accessory’s points to the target, deform the rest minimally.',
    image: '/help/fit.webp',
    pickIf: 'A rigid piece must land on exact points and otherwise keep its shape.',
    help: {
      what: 'Purely geometric point attachment: pins the source’s marker vertices exactly onto the target’s marker points and deforms the bulk minimally. NO body awareness — no surface grip, contact field, collision, or layer handling. You control the attachment entirely through the marker pairs you place.',
      when: 'A rigid-ish accessory that attaches at a few known points and must keep its shape (glasses → face, a badge, a prop). If the piece should conform to and interact with the body surface — drape, grip a seam, hide the body under it — use Refit instead.',
      markers: 'Required — this is the whole mechanism. Pairs around the contact ring; every disconnected island needs at least one (ideally 3+).',
    },
    steps: [
      refsStep(
        'the accessory to attach',
        'the mesh it attaches to',
        'Fit moves the accessory’s marker vertices exactly onto the target’s marker points — nothing else pulls it.',
      ),
      markersStep({
        title: 'Mark every attachment point',
        count: 'At least 3 pairs, and at least one — ideally 3+ — on EVERY disconnected piece. A loose part with no marker will not move at all.',
        where: 'Place them around the ring where the accessory touches the target: the rim of glasses on the temples and nose bridge, the base of a horn, the back of a badge.',
        why: 'Markers are the entire mechanism here — there is no surface snapping to fall back on.',
      }),
      {
        key: 'settings',
        label: 'Fit',
        title: 'Set rigidity and attach',
        how: [
          'Rigidity high (toward 10⁰) = the accessory keeps its shape and only the area near the anchors bends.',
          'Rigidity low (toward 10⁻⁴) = it stretches more freely to reach the anchors.',
          'Start high for a solid prop, lower it only if the anchors are visibly not being reached.',
          'Press Fit.',
        ],
      },
      resultStep('the attached accessory as .obj'),
    ],
  },
]

export function toolMeta(tool: ToolKind | null): ToolMeta | undefined {
  return TOOLS.find((t) => t.key === tool)
}

function BeforeAfter({ meta }: { meta: ToolMeta }) {
  // Show the art if the file exists; if it 404s (not added yet) fall back to
  // the placeholder instead of a broken-image icon. So dropping a webp into
  // public/help/ is enough to make it appear — no code change needed.
  const [failed, setFailed] = useState(false)
  if (meta.image && !failed) {
    return (
      <img
        className="tool-card-img"
        src={meta.image}
        alt={`${meta.label} before and after`}
        onError={() => setFailed(true)}
      />
    )
  }
  // Placeholder until real art exists.
  return (
    <div className="tool-card-img placeholder" aria-hidden>
      <span>before</span>
      <span className="arrow">→</span>
      <span>after</span>
    </div>
  )
}

export function ToolSelect({ onPick }: { onPick: (t: ToolKind) => void }) {
  return (
    <div className="tool-select">
      <div className="tool-select-head">
        <h1>What do you want to do?</h1>
        <p className="subtle">
          Pick a tool and it walks you through it step by step. You can switch at any time —
          uploads and markers stay.
        </p>
      </div>
      <div className="tool-grid">
        {TOOLS.map((t) => (
          // The whole card is the button: a 220px target beats a 90px one, and
          // there is nothing else inside it to click.
          <button
            key={t.key}
            type="button"
            className="tool-card"
            aria-label={`Start ${t.label} — ${t.tagline}`}
            onClick={() => onPick(t.key)}
          >
            {t.beta && <span className="beta-badge">beta</span>}
            <BeforeAfter meta={t} />
            <h3>{t.label}</h3>
            <p className="subtle">{t.tagline}</p>
            <p className="tool-card-pick">
              <b>Pick this if:</b> {t.pickIf}
            </p>
            <span className="tool-card-cta">Start →</span>
          </button>
        ))}
      </div>
    </div>
  )
}

/**
 * The "what do I do now" card the sidebar shows above the active step's
 * controls. This is the guide living next to the tool, not in a manual.
 */
export function StepGuide({ step, index, total }: { step: StepMeta; index: number; total: number }) {
  return (
    <section className="panel step-guide">
      <div className="step-guide-head">
        <span className="step-guide-num">Step {index + 1}/{total}</span>
        {step.optional && <span className="step-guide-optional">optional</span>}
      </div>
      <h3>{step.title}</h3>
      <ol className="step-guide-list">
        {step.how.map((line, i) => (
          <li key={i}>{line}</li>
        ))}
      </ol>
      {step.why && <p className="step-guide-why">{step.why}</p>}
    </section>
  )
}

/**
 * "?" bubble next to the tool title. Opens on hover, stays open when clicked.
 *
 * Positioned FIXED from the button's rect and clamped to the viewport: the
 * sidebar sits at the right edge of the window, so an absolutely-positioned
 * popover anchored to its left ran off-screen.
 */
export function HelpTip({ tool }: { tool: ToolKind }) {
  const [open, setOpen] = useState(false)
  const [pinned, setPinned] = useState(false)
  const [pos, setPos] = useState<{ left: number; top: number; maxHeight: number } | null>(null)
  const wrapRef = useRef<HTMLSpanElement>(null)
  const btnRef = useRef<HTMLButtonElement>(null)
  const closeTimer = useRef<number | undefined>(undefined)
  const meta = TOOLS.find((t) => t.key === tool)

  const POP_WIDTH = 320
  const MARGIN = 12

  const place = useCallback(() => {
    const r = btnRef.current?.getBoundingClientRect()
    if (!r) return
    const left = Math.max(
      MARGIN,
      Math.min(r.left, window.innerWidth - POP_WIDTH - MARGIN),
    )
    const below = window.innerHeight - r.bottom - MARGIN * 2
    const above = r.top - MARGIN * 2
    // Prefer below; flip up only when below is cramped and above is roomier.
    if (below < 220 && above > below) {
      setPos({ left, top: Math.max(MARGIN, r.top - MARGIN - above), maxHeight: above })
    } else {
      setPos({ left, top: r.bottom + 8, maxHeight: below })
    }
  }, [])

  useLayoutEffect(() => {
    if (open) place()
  }, [open, place])

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false)
        setPinned(false)
      }
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpen(false)
        setPinned(false)
      }
    }
    window.addEventListener('mousedown', onDown)
    window.addEventListener('keydown', onKey)
    window.addEventListener('resize', place)
    // Capture phase: the sidebar scrolls, and it is not the event target.
    window.addEventListener('scroll', place, true)
    return () => {
      window.removeEventListener('mousedown', onDown)
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('resize', place)
      window.removeEventListener('scroll', place, true)
    }
  }, [open, place])

  useEffect(() => () => window.clearTimeout(closeTimer.current), [])

  if (!meta) return null

  const openNow = () => {
    window.clearTimeout(closeTimer.current)
    setOpen(true)
  }
  const closeSoon = () => {
    if (pinned) return
    // Small grace period so the pointer can travel from the button to the
    // popover without it vanishing under the cursor.
    closeTimer.current = window.setTimeout(() => setOpen(false), 160)
  }

  return (
    <span className="helptip" ref={wrapRef} onMouseEnter={openNow} onMouseLeave={closeSoon}>
      <button
        ref={btnRef}
        className="helptip-btn"
        aria-label={`Help for ${meta.label}`}
        aria-expanded={open}
        onClick={() => {
          if (open && pinned) {
            setOpen(false)
            setPinned(false)
          } else {
            setPinned(true)
            setOpen(true)
          }
        }}
        onFocus={openNow}
        onBlur={closeSoon}
      >
        ?
      </button>
      {open && pos && (
        <div
          className="helptip-pop"
          role="dialog"
          style={{ left: pos.left, top: pos.top, width: POP_WIDTH, maxHeight: pos.maxHeight }}
        >
          <h4>{meta.label}</h4>
          <p><b>What:</b> {meta.help.what}</p>
          <p><b>When:</b> {meta.help.when}</p>
          <p><b>Markers:</b> {meta.help.markers}</p>
        </div>
      )}
    </span>
  )
}
