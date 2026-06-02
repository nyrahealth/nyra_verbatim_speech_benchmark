# Normalization Policy

The benchmark separates raw model output from normalized benchmark input.

```text
raw model transcript
  -> fixed raw prediction schema
  -> global normalizer + optional source adapter
  -> normalized prediction JSONL
  -> evaluator
```

## Global Normalizer

The global normalizer should be conservative and idempotent. Applying it to an
already canonical transcript should not change its meaning.

Good global rules:

- `[uh] -> [UH]`, `[um] -> [UM]`
- bare English `uh` / `um` -> `[UH]` / `[UM]`
- German `äh` / `ähm` -> `[UH]` / `[UM]`
- `[laugh]`, `[laughs]`, `[laughing]` -> `[laughter]`
- `[clears throat]`, `[throat clear]` -> `[throatclearing]`
- Unicode cutoff dashes attached to word fragments -> `-`
- simple digit forms -> spoken words

Avoid global rules that can change normal language into a disfluency tag.

## Source Adapters

Adapters run before the global normalizer. They are for provider-specific quirks
that are not safe across all models.

Current adapter:

- `assemblyai`: maps laughter-like strings such as `haha`, `hehe`, and `ah` to
  `[laughter]` because this behavior was observed in AssemblyAI outputs. The
  bare `ah` rule is intentionally not global.

## Cutoff Ambiguity

Cutoffs are not always spelled consistently by models or annotators. A model may
write the same interrupted word as `w*`, `w-`, `w—`, `w–`, or use another
dash-like symbol.

The benchmark normalizes common marker variants to the internal `-` marker, then
reports multiple metrics:

- `cutoff_f1`: prediction must contain a cutoff marker.
- `cutoff_strict_f1`: aligned cutoff stem must match.
- `cutoff_lenient_f1`: any aligned prediction at the cutoff position counts as
  coverage, while hallucinated cutoff markers are still penalized.
- `cutoff_lenient_recall`: only asks whether gold cutoff positions were covered.

This makes cutoff evaluation transparent instead of pretending there is only one
unambiguous spelling.
