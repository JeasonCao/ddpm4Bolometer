# Research collaboration

User instructions take precedence. Preserve current scientific methods unless a change is authorized.

## Code placement

- `src/`: reusable algorithms/models; `scripts/`: stable task entry points.
- `analysis/`: result processing and plots; record datasets and selection rules.
- `experiments/<topic>/`: hypothesis-driven studies, orchestration and configs. README records question, inputs, settings, outputs, conclusions and limitations. Committing an experiment does not make it a default/recommended method.
- `archive/`: historical studies. Explain stale assumptions and environment dependencies; do not silently change scientific behavior.
- `scratch/`: disposable exploration, ignored by Git. Before relying on its result as evidence, organize code and provenance into experiments/analysis.
- `tests/`: small meaningful checks; do not run long jobs merely to test directory changes.

Decide placement by purpose, not whether an agent or human wrote it. Reuse existing code. Modify canonical implementations, not compatibility entry files. Promotion to a default scientific method requires evidence and user agreement.

## Storage

Waveforms, weights, environments, logs and bulk results stay outside Git. The data disk stores waveforms/weights; Obsidian stores the project diary, unique TODO and the established results/figure directories. Small intentional configurations and evidence summaries can be versioned; do not blanket-ignore all JSON.

Inputs are read-only. Record source file/row identity, units, settings and hashes. Use independent output paths. New code should configure machine paths through arguments or environment variables.

## Workflow

Update diary for substantive work and the unique TODO for actions. Distinguish implementation checks from scientific validation. Solver success/file existence alone does not establish research validity.

Commit/push only when authorized. Review explicit staged paths; avoid `git add .`. Do not commit secrets, installed dependencies, caches, logs or data. Keep scientific changes separate from organization. Long GPU/cloud/full runs require existing authorization. Do not contact collaborators unless explicitly asked.

For 2026-11-10, user authorized two local commits and will push manually. No upstream fetch/pull/merge is authorized. Inventory/mapping/limitations are in docs/reorganization_20261110/.
