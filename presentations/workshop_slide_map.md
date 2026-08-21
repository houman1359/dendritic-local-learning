# Workshop deck — slide maps and cut plans

Deck: `dendritic_credit_workshop.pdf` — one 42-page build: 35 core frames
(footer `n / 35`), a Backup divider (p.36) and backup frames A1–A6
(pp.37–42). Slide numbers below are the core frame numbers printed in the
footer. Narration and timings live in `speaker_notes_workshop.md`.

Numbers policy: content slides print at most one large headline number; every
secondary statistic was relocated into the narration of its slide in
`speaker_notes_workshop.md` (marked there in bold). Whichever cut you
present, the notes are the only place those numbers exist — carry them.

Frame index: 1 title · 2 hook (one error, ~10^4 synapses) · 3 credit ·
4 BP reference · 5 divider · 6 taxonomy · 7 per-neuron coordinate ·
8 this talk · 9 divider · 10 forward quotient · 11 eligibility ·
12 factorization/transport · 13 four-rung ladder · 14 credit operator U(M) ·
15 capture/Ky–Fan · 16 interior optimum · 17 route gain · 18 divider ·
19 conventions · 20 bandwidth · 21 ownership · 22 address factorial ·
23 depth/alignment dose · 24 physical depth · 25 real arbors ·
26 focal shunt · 27 conductance boundary · 28 alignment rescue ·
29 animal coordinates · 30 divider · 31 phase plane · 32 four inequalities ·
33 wiring economics · 34 predictions · 35 four answers.

---

## 15-minute conceptual (15 frames)

**Keep:** 1, 2, 3, 4, 7, 8 · 10, 11, 12, 14 · 20, 22, 26 · 31, 35.
Theory act compressed to slides 10–12 + 14; one experiment per ladder rung
(20 coordinate, 22 address, 26 gain); close on the phase plane and the
four answers. Pace: ~58 s per frame; give 12 and 14 their full time and take
it from 20 and 26. In this cut, speak only each kept slide's headline number
plus its strongest relocated statistic from the notes (12: autodiff check;
20: per-task ranges; 22: the K-ladder; 26: the factor freeze) — there is no
time for the rest.

What each cut loses:

- **5, 9, 18, 30 (dividers)** — loses the ladder-progress motif and the
  breathing room between acts; replace each with one spoken beat.
- **6 (taxonomy)** — loses the four-family map with citations; name-check
  "modulators, random feedback, state inference, error compartments" while
  on slide 7.
- **13 (ladder map)** — loses the visual four-rung map of the talk; speak
  the ladder (scalar, coordinate, address, gain) as you land on 14.
- **15 (capture/Ky–Fan)** — loses the optimality criterion and the +0.55
  spectral number (full-alignment condition); assert "anatomy is optimal
  when task covariance is coarse-dominant in the tree's basis" in one
  sentence on 14.
- **16 (interior optimum)** — loses the prediction stated *before* the data;
  slide 22's crossover must carry it retroactively ("an interior optimum,
  where the theory put it").
- **17 (route gain theory)** — loses the reliability weight a\* and the
  Sherman–Morrison identity, so slide 26 becomes phenomenology; say "a shunt
  is a rank-one edit of the transport operator" verbally on 26.
- **19 (conventions)** — loses the stated matched-budget / paired-seed /
  point-emulation discipline; compress to one sentence before 20 ("everything
  is matched-resource, paired by seed, and a point network given the same
  fields matches by construction").
- **21 (ownership)** — loses the bandwidth-vs-map dissociation (the
  fractions-of-a-point control).
- **23 (depth/alignment dose)** — loses the steep alignment dose response
  (+27.7 pp final step) and the D_r = H validation.
- **24 (physical depth)** — loses the +31 pp constructed existence proof
  *and* its paired negative — the strongest "conditional, never free" slide.
- **25 (real arbors)** — loses the MICrONS wiring-efficiency result (85% of
  dense capture at 7% of wiring).
- **27 (conductance boundary)** — loses the state-dependence boundary that
  makes the shunt claim falsifiable; flag it in one clause on 26.
- **28 (alignment rescue)** — biggest casualty: loses the measured-null →
  rescue arc; when on 31, point at the gray × and say "measured responses
  are a predicted null; imposed alignment rescues the same dictionary."
- **29 (animal coordinates)** — loses the in-vivo signed-coordinate evidence.
- **32 (four inequalities)** — loses the compact when-does-local-win summary.
- **33 (wiring economics)** — loses the counting argument for why dendrites
  at all; fold its one-liner into 34 if the audience is neuro.
- **34 (predictions)** — only cut for a pure-ML room; otherwise it stays and
  35 absorbs its closing beat.

---

## 30-minute default (35 frames, as built)

Present `dendritic_credit_workshop.pdf` linearly; timings, per-act
"if pressed" lines and every relocated statistic are in
`speaker_notes_workshop.md`. The Backup section (divider p.36, frames A1–A6
pp.37–42 of the same PDF) stays in reserve for Q&A — jump table at the end
of the notes, including two verbal answers (Sherman–Morrison in full,
full-tree all-scan boundary) whose numbers appear nowhere in the compiled
deck.

Nothing is cut, so the only risk is pace: 30 content frames in 30 minutes.
The act-level budget is the contract — if an act closes >45 s late, take its
"if pressed" cut on the spot rather than rushing the phase plane at the end.
Cheapest skims if needed: 21 and 27 (each collapses to one spoken sentence,
with A3/A5 backup available).

---

## 40-minute with backup promotions (40 frames)

Present the deck linearly and jump to the promoted backup page at each
marked point, then return to the next core frame. Promotions add ~8 minutes;
the remaining ~2 minutes go to slower dwell on 12, 14, 22, 24 and 31.

| After core frame | Jump to | What the promotion adds |
|---|---|---|
| 12 (factorization) | A1 (p.37) | the adjoint chain in full — implicit function to path product, Almeida–Pineda lineage, when the transpose matters |
| 14 (credit operator) | A2 (p.38) | the U(M) derivation with its exactness caveat, plus the rho 0.937/0.916 predictive-validity numbers |
| 17 (route gain) | A5 (p.41) | the trained-model test of a\*: one-step ordering holds (49/50), final-loss difference null — plus same-span bias–variance conditioning |
| 22 (address factorial) | A3 (p.39) | the factorial control battery and exact point-implementation equivalence — the "no dendritic magic" receipt |
| 24 (physical depth) | A4 (p.40) | the outcome-independent validity audit (840/1,840 runs excluded) and exact-vs-BP average agreement |

Kept as backup only: **A6** (p.42, references). Five further frames
(quotient-rule steps, Ky–Fan/weighted capture, Sherman–Morrison selectivity,
same-span bias–variance in full, full-tree all-scan boundary) are preserved
uncompiled in the `\iffalse` block of
`dendritic_credit_workshop_appendix.tex`; restore one by moving it above the
block if the audience warrants it.

Footer caveat: backup pages are numbered A1–A6 and the core footer says
"/35", so promoted frames are visibly backup material — say "let me pull one
backup slide" and the room reads it correctly.
