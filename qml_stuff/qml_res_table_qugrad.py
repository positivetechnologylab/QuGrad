#!/usr/bin/env python3
import re
import sys

def parse_log(path):
    """
    Parse a log file for accuracy and macro-avg precision, recall, f1.
    Returns a dict with keys: 'accuracy', 'precision', 'recall', 'f1'.
    """
    try:
        text = open(path).read()
    except FileNotFoundError:
        print(f"Error: Could not find log file: {path}")
        sys.exit(1)

    metrics = {}
    
    # Try to find summary mean accuracy first (more robust if individual folds vary)
    # Looking for "mean ± std = 0.764 ± 0.043" type line
    m_mean = re.search(r'mean\s*±\s*std\s*=\s*([0-9]*\.?[0-9]+)', text)
    if m_mean:
        metrics['accuracy'] = float(m_mean.group(1))
    else:
        # Fallback to last "accuracy =" found, but typically we want the mean
        # Let's check if there's a specific summary block or we just average folds manually
        # This regex picks the LAST accuracy if multiple are present, which might be a fold accuracy
        # Better to look for the "mean" line which is standard in these scripts.
        pass

    # If the script prints "mean ± std", we use that for accuracy. 
    # For precision/recall/f1, these scripts usually print report per fold.
    # We should probably average them if the script doesn't print a global average.
    # However, standard classification_report only prints per-fold in the loop.
    # Let's parse all "macro avg" lines and average them.
    
    precisions = []
    recalls = []
    f1s = []
    
    # Iterate over all matches of macro avg
    # macro avg      0.826     0.824     0.825        80
    pattern = r'macro\s+avg\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)'
    for m in re.finditer(pattern, text):
        precisions.append(float(m.group(1)))
        recalls.append(float(m.group(2)))
        f1s.append(float(m.group(3)))
        
    if precisions:
        metrics['precision'] = sum(precisions) / len(precisions)
        metrics['recall']    = sum(recalls)    / len(recalls)
        metrics['f1']        = sum(f1s)        / len(f1s)
    
    # If accuracy wasn't found in summary, try to average fold accuracies
    if 'accuracy' not in metrics:
        # accuracy = 0.825
        accs = re.findall(r'accuracy\s*=\s*([0-9]*\.?[0-9]+)', text)
        if accs:
            acc_vals = [float(x) for x in accs]
            metrics['accuracy'] = sum(acc_vals) / len(acc_vals)
            
    return metrics

def main():
    # If arguments are provided, expect 5 logs. Otherwise default to known files.
    if len(sys.argv) > 1:
        if len(sys.argv) != 6:
            print("Usage: python qml_res_table_qugrad.py ideal_old noisy_old ideal_new noisy_new baseline")
            print("Or run without arguments to use default filenames.")
            sys.exit(1)
        ideal_old, noisy_old, ideal_new, noisy_new, baseline_log = sys.argv[1:6]
    else:
        # Default files
        ideal_old    = "dualann_qugrad_ideal.log"
        noisy_old    = "dualann_qugrad_noisy.log"
        ideal_new    = "dualann_qugrad_ideal_new.log"
        noisy_new    = "dualann_qugrad_noisy_new.log"
        baseline_log = "logreg_res.log"

    print(f"Processing logs:\n  Old Ideal: {ideal_old}\n  Old Noisy: {noisy_old}\n  New Ideal: {ideal_new}\n  New Noisy: {noisy_new}\n  Baseline:  {baseline_log}\n")

    m_ideal_old = parse_log(ideal_old)
    m_noisy_old = parse_log(noisy_old)
    m_ideal_new = parse_log(ideal_new)
    m_noisy_new = parse_log(noisy_new)
    m_baseline  = parse_log(baseline_log)

    runs = [
        ('QuGrad Ideal (Old)', m_ideal_old),
        ('QuGrad Noisy (Old)', m_noisy_old),
        ('QuGrad Ideal (New)', m_ideal_new),
        ('QuGrad Noisy (New)', m_noisy_new),
        ('Baseline', m_baseline)
    ]

    for name, d in runs:
        missing = [k for k in ('accuracy','precision','recall','f1') if k not in d]
        if missing:
            print(f"Error: {name} log missing {missing}")
            # Don't exit immediately, try to show what we have

    # Calculate scaled metrics (percentage relative to baseline)
    def safely_scale(num, denom):
        if denom == 0 or num is None or denom is None:
            return 0.0
        return (num / denom) * 100

    def get_scaled_metrics(metrics, baseline):
        return {
            'accuracy':  safely_scale(metrics.get('accuracy'), baseline.get('accuracy')),
            'precision': safely_scale(metrics.get('precision'), baseline.get('precision')),
            'recall':    safely_scale(metrics.get('recall'), baseline.get('recall')),
            'f1':        safely_scale(metrics.get('f1'), baseline.get('f1'))
        }

    sc_ideal_old = get_scaled_metrics(m_ideal_old, m_baseline)
    sc_noisy_old = get_scaled_metrics(m_noisy_old, m_baseline)
    sc_ideal_new = get_scaled_metrics(m_ideal_new, m_baseline)
    sc_noisy_new = get_scaled_metrics(m_noisy_new, m_baseline)

    table = f"""\\begin{{table}}[h!]
\\centering
\\caption{{Performance of dual-annealing (QuGrad Ideal vs. Noisy, Old vs. New) scaled to a classical logistic-regression baseline (100\\%).}}
\\label{{tab:dualann_qugrad_full_comparison}}
\\vspace{{2mm}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Run}} & \\textbf{{Accuracy (\\%)}} & \\textbf{{Precision (\\%)}} & \\textbf{{Recall (\\%)}} & \\textbf{{Macro F1 (\\%)}} \\\\
\\midrule
QuGrad Ideal (Old) & {sc_ideal_old['accuracy']:.1f} & {sc_ideal_old['precision']:.1f} & {sc_ideal_old['recall']:.1f} & {sc_ideal_old['f1']:.1f} \\\\
QuGrad Noisy (Old) & {sc_noisy_old['accuracy']:.1f} & {sc_noisy_old['precision']:.1f} & {sc_noisy_old['recall']:.1f} & {sc_noisy_old['f1']:.1f} \\\\
QuGrad Ideal (New) & {sc_ideal_new['accuracy']:.1f} & {sc_ideal_new['precision']:.1f} & {sc_ideal_new['recall']:.1f} & {sc_ideal_new['f1']:.1f} \\\\
QuGrad Noisy (New) & {sc_noisy_new['accuracy']:.1f} & {sc_noisy_new['precision']:.1f} & {sc_noisy_new['recall']:.1f} & {sc_noisy_new['f1']:.1f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    print(table)

if __name__ == "__main__":
    main()

