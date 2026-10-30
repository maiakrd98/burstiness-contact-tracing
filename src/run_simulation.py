import argparse
import numpy as np
import contact_process
import infection_process

def main():
    parser = argparse.ArgumentParser(
                        prog='run_simulation',
                        description='Run one index case simulation with '
                                    'given parameters')

    parser.add_argument('--d',
                        type=int,
                        help='number of contacts of the index case',
                        required=True)
    
    parser.add_argument('--lam',
                        type=float,
                        help='turnover rate of contacts',
                        required=True)

    parser.add_argument('--t_max',
                        type=float,
                        help='time to run simulation until',
                        required=True)

    parser.add_argument('--alpha',
                        type=float,
                        help='shape parameter of overall generation '
                             'time distribution',
                        required=True)

    parser.add_argument('--beta',
                        type=float,
                        help='rate parameter of overall generation '
                             'time distribution',
                        required=True)

    parser.add_argument('--psi',
                        type=float,
                        help='burstiness  parameter',
                        required=True)

    parser.add_argument('--b',
                        type=float,
                        help='infectiousness tuning parameter',
                        required=True)

    parser.add_argument('--seed',
                        type=int,
                        help='random seed',
                        required=True)

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    df = contact_process.contact_simulation(args.d, args.lam, args.t_max, rng=rng)

    new_df = infection_process.sim_one_index_case(df, args.alpha, args.beta, args.psi, args.b, rng=rng)
    print(new_df)


if __name__ == "__main__":
    main()