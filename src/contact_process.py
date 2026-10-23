import numpy as np
import pandas as pd

def contact_process(d, lam, t_max, seed=None):
    # add comment to explain function and necessary parts

    # random number generator
    rng = np.random.default_rng(seed)

    # track the number of contacts, constant d, of node 0
    slot_id = list(range(1, d + 1)) # slot_id[k] = id of node in slot k
    slot_start = [0.0] * d # slot_start[k] = time node starts contact with node 0

    # assign id to new nodes
    next_id = d + 1

    # initialize storage for finished contacts and their id, start contact time, and end contact time
    contacts = []

    # initalize simulation time
    t = 0.0

    # repeated loop until t_max is reached
    while True:
        # determine time until next turnover among d contacts of node 0
        t += rng.exponential(1 / lam)

        # exit loop when turnover falls beyond t_max
        if t >= t_max:
            break

        # randomly pick which contact slot turnsover
        k = rng.integers(d)

        # record which contact is leaving
        contacts.append((slot_id[k], slot_start[k], t))

        # replace contact with new node and record start contact time and id
        slot_id[k] = next_id
        slot_start[k] = t
        next_id += 1

    # record remaining contacts after t_max is reached
    contacts += [(slot_id[k], slot_start[k], t_max) for k in range(d)]

    # build output for infection model
    df = pd.DataFrame(contacts, columns=["id", "start_contact", "end_contact"])
    return df.sort_values("id").reset_index(drop=True)
        