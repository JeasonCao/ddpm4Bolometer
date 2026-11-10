"""Export one row per trigger from an Octopus Processed file to CSV (runs inside the Octopus container).

  python3 export_of_table.py <Processed_*.root> <out.csv>
"""
import sys, csv
import ROOT

COLS = [('module', 'triggersample'), ('module', 'isnoise'), ('module', 'issignal'),
        ('numberoftriggers', 'numberoftriggers'), ('baseline', 'RMS'), ('baselineslope', 'slope'),
        ('maxminusbaseline', 'amplitude'), ('triggerdelay', 'delay'),
        ('optimumfilter', 'amplitude'), ('optimumfilter', 'delay'), ('optimumfilter', 'Chi2'),
        ('optimumfilter', 'Chi2AtLowerLimit'), ('optimumfilter', 'Chi2AtUpperLimit'), ('optimumfilter', 'good')]

f = ROOT.TFile.Open(sys.argv[1])
trees = {t: f.Get(t) for t in {t for t, _ in COLS}}
n = trees['module'].GetEntries()
assert all(t.GetEntries() == n for t in trees.values())
with open(sys.argv[2], 'w', newline='') as out:
    w = csv.writer(out)
    w.writerow([f'{t}_{b}' for t, b in COLS])
    for i in range(n):
        for t in trees.values():
            t.GetEntry(i)
        w.writerow([getattr(trees[t], b) for t, b in COLS])
print(f'{n} rows -> {sys.argv[2]}')
