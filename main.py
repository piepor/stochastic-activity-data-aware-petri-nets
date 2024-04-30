import argparse
import running_example
import method_evaluation
import method_analysis


def main(mode='running-example', compute=False):
    if mode == 'running-example':
        running_example.implementation(compute)
    elif mode == 'method-evaluation':
        method_evaluation.implementation(compute) 
    elif mode == 'method-analysis':
        method_analysis.implementation(compute)
    else:
        raise NotImplementedError('Mode not implemented')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['running-example', 'method-evaluation'])
    parser.add_argument('--compute', type=bool, default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    mode = args.mode
    compute = args.compute
    main(mode, compute)
