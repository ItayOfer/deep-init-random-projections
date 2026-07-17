# Brief — <slug> (<yyyy-mm-dd>)

> Copy this template to `<yyyy-mm-dd>_<slug>.md`. A brief must be **self-sufficient**: a fresh agent with zero conversation history should be able to execute from it plus the onboarding chain.

**Onboarding chain (read in order before starting):** `README.md` → `docs/RESEARCH_LOG.md` → `docs/plans_handoffs/FRONTIER.md` → this brief → `CLAUDE.md` (conventions). Task-specific deep dives listed under Context below.

## Goal

One sentence: the question this task answers or the artifact it produces.

## Context

- Why now — which finding/decision triggered this (link the campaign README / FRONTIER row).
- Prior art in-repo — what exists that this builds on (files, results, scripts). Never re-derive what a linked doc already establishes.

## Deliverables

Numbered, concrete, checkable. For experiments: which JSONs must exist in `reports/results/`, which README documents the outcome. For proofs/writing: which .tex/.md changes, building where.

## Constraints

- Branch: `main` (additive only) / `work/<slug>` (touches shared files) — pick one.
- Standing rules that apply (seed 42, no clipping, naming, pass criterion, width conventions…).
- Budget hints: cluster wall-time, local-vs-cluster, what NOT to touch.

## Definition of done

The falsifiable checklist: results pulled + committed, README updated, FRONTIER row updated, verification note (which numbers were re-checked against which files).

## Outcome  *(filled by the worker at the end)*

What happened, with numbers and file pointers. Surprises and deviations from the brief. What the oracle should verify.
