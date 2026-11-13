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

## Standing commit authorization (2026-11-13)

The user authorizes automatic local commits after each coherent task's code changes are finished and appropriate checks pass. Do not ask again solely for permission to commit. A commit should represent a completed change, not each individual file save.

Include only reviewed code and documentation for the current task; leave other tasks' changes untouched. Do not treat pending user decisions or failed/incomplete checks as ready for a completion commit. Preserve the explicit-path staging and data-exclusion rules above.

Record the change and validation in the project diary, and report the commit hash. Push remains manual unless separately requested. This standing authorization does not authorize upstream fetch/pull/merge or additional experiments.

For 2026-11-10, user authorized two local commits and will push manually. No upstream fetch/pull/merge is authorized. Inventory/mapping/limitations are in docs/reorganization_20261110/.
