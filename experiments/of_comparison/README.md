# OF / DDPM comparison

Canonical former scripts/of_compare workflow. Old Python/shell entries forward here; the old TOML template remains for compatibility.

Question: compare OF and DDPM on matching events. Generate continuous noise/events, export old Octopus bins, cut DDPM windows, reconstruct OF and compare amplitudes. Internal paths use this new directory; scientific behavior is unchanged.

Run from repo root. Read-only example: `python -m experiments.of_comparison.generate_of_stream --help`. Inspect commands before execution: generation can be large and Octopus needs its established external environment. Existing fixed data paths were not broadly refactored.

Data disk: OFcompare_*, eval_data_OFcompare, OFcompare_ddpm. Figures follow existing Obsidian results mapping. Pilot runs 999996/999997; full runs 999998/999999. Independent window OF remains scripts/window_of.py.

AP/ANPS choice, oracle controls, per-peak calibration and event splits differ by study. Do not mix results without matching definitions. No long run was executed in this cleanup.
