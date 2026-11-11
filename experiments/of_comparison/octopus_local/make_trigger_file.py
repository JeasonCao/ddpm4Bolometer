"""Write an Octopus (old version) trigger file with our own trigger list.

All metadata (TParameters, channel lists, bad_interval_tree) are copied from a trigger file that
Octopus itself wrote for the same run; only trigger_tree is replaced.

  ideal: signal triggers at the true onsets (TriggerSample = onset_sample, so the 10 s window with
         1.5 s pretrigger is exactly the DDPM window), noise triggers at the reserved quiet periods.

Runs inside the Octopus container (PyROOT from /software/root_v6.32.08):
  python3 make_trigger_file.py --template Trigger_Run_999996_Channel_1.root --truth <truth dir> --out <file>
"""

# Repository entry bootstrap; no data or model is loaded here.
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_ROOT = _RepoPath(__file__).resolve().parents[3]
_repo_sys.path.insert(0, str(_REPO_ROOT))
_repo_sys.path.insert(0, str(_REPO_ROOT / "scripts"))
import argparse, csv
import ROOT
from array import array


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--template', required=True)
    ap.add_argument('--truth', required=True, help='run truth/ directory with events.csv and noise_windows.csv')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    trig = [(int(r['onset_sample']), True) for r in csv.DictReader(open(f'{a.truth}/events.csv'))]
    trig += [(int(r['trigger_sample']), False) for r in csv.DictReader(open(f'{a.truth}/noise_windows.csv'))]
    trig.sort()

    fin = ROOT.TFile.Open(a.template)
    fout = ROOT.TFile.Open(a.out, 'RECREATE')
    for key in fin.GetListOfKeys():
        if key.GetName() == 'trigger_tree':
            continue
        obj = key.ReadObj()
        fout.cd()
        if obj.InheritsFrom('TTree'):
            obj.CloneTree(-1).Write()
        else:
            obj.Write(key.GetName())

    fout.cd()
    t = ROOT.TTree('trigger_tree', 'trigger_tree')
    ts, rms = array('Q', [0]), array('d', [0.])
    pul, npul, mu, isn, iss = array('b', [0]), array('i', [-1]), array('b', [0]), array('b', [0]), array('b', [0])
    t.Branch('TriggerSample', ts, 'TriggerSample/l')
    t.Branch('TriggerRMSRatio', rms, 'TriggerRMSRatio/D')
    t.Branch('IsPulser', pul, 'IsPulser/O')
    t.Branch('NPulser', npul, 'NPulser/I')
    t.Branch('IsMuon', mu, 'IsMuon/O')
    t.Branch('IsNoise', isn, 'IsNoise/O')
    t.Branch('IsSignal', iss, 'IsSignal/O')
    for s, sig in trig:
        ts[0] = s; iss[0] = sig; isn[0] = not sig
        t.Fill()
    t.Write()
    fout.Close()
    print(f'wrote {a.out}: {sum(s for _, s in trig)} signal, {sum(not s for _, s in trig)} noise triggers')


if __name__ == '__main__':
    main()
