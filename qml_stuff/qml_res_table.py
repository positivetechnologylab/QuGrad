#!/usr/bin/env python3
import re
import sys

def parse_log(path):
    """
    Parse a log file for accuracy and macro-avg precision, recall, f1.
    Returns a dict with keys: 'accuracy', 'precision', 'recall', 'f1'.
    """
    text = open(path).read()
    metrics = {}
    m = re.search(r'accuracy\s*=\s*([0-9]*\.?[0-9]+)', text)
    if m:
        metrics['accuracy'] = float(m.group(1))
    # macro avg    precision recall f1-score
    m = re.search(
        r'macro\s+avg\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)',
        text
    )
    if m:
        metrics['precision'] = float(m.group(1))
        metrics['recall']    = float(m.group(2))
        metrics['f1']        = float(m.group(3))
    return metrics

def main():
    if len(sys.argv) != 4:
        print("Usage: python qml_res_table.py dualann_ideal.log dualann_noisy.log logreg_result.log")
        sys.exit(1)

    ideal_log, noisy_log, baseline_log = sys.argv[1], sys.argv[2], sys.argv[3]

    m_ideal    = parse_log(ideal_log)
    m_noisy    = parse_log(noisy_log)
    m_baseline = parse_log(baseline_log)

    for name, d in [('Ideal', m_ideal), ('Noisy', m_noisy), ('Baseline', m_baseline)]:
        missing = [k for k in ('accuracy','precision','recall','f1') if k not in d]
        if missing:
            print(f"Error: {name} log missing {missing}")
            sys.exit(1)

    sc_ideal = {
        'accuracy':  m_ideal['accuracy']   / m_baseline['accuracy']   * 100,
        'precision': m_ideal['precision']  / m_baseline['precision']  * 100,
        'recall':    m_ideal['recall']     / m_baseline['recall']     * 100,
        'f1':        m_ideal['f1']         / m_baseline['f1']         * 100
    }
    sc_noisy = {
        'accuracy':  m_noisy['accuracy']   / m_baseline['accuracy']   * 100,
        'precision': m_noisy['precision']  / m_baseline['precision']  * 100,
        'recall':    m_noisy['recall']     / m_baseline['recall']     * 100,
        'f1':        m_noisy['f1']         / m_baseline['f1']         * 100
    }

    table = f"""\\begin{{table}}[h!]
\\centering
\\caption{{Performance of dual-annealing (Ideal vs. Noisy) scaled to a classical logistic-regression baseline (100\\%).}}
\\label{{tab:dualann_comparison}}
\\vspace{{2mm}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Run}} & \\textbf{{Accuracy (\\%)}} & \\textbf{{Precision (\\%)}} & \\textbf{{Recall (\\%)}} & \\textbf{{Macro F1 (\\%)}} \\\\
\\midrule
Ideal & {sc_ideal['accuracy']:.1f} & {sc_ideal['precision']:.1f} & {sc_ideal['recall']:.1f} & {sc_ideal['f1']:.1f} \\\\
Noisy & {sc_noisy['accuracy']:.1f} & {sc_noisy['precision']:.1f} & {sc_noisy['recall']:.1f} & {sc_noisy['f1']:.1f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    print(table)

if __name__ == "__main__":
    main()

