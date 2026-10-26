import numpy as np
import pandas as pd

def contact_simulation(d, lam, t_max, seed=0):
    """Simulate the contact process for the index case of node 0.

    Parameters
    ----------
    d : int
        Number of simultaneous contacts of node 0. 
        This must be a positive integer.
    lam : float
        Turnover rate (lambda) represents the rate at which contacts are replaced by new contacts.
        This must be a positive number.
    t_max : float
        Length of time to simulate the contacts for and the time the simulation is stopped.
        This must be a positive number.
    seed : int (optional)
        Seed for the random number generator (default of 0).

    Returns
    -------
    df = pandas.DataFrame
        Data frame with one row per contact, sorted by id number, with columns:

        id : int
            Id number of the contact node. Node 0, the index case, does not have a row. The initial
            contacts' id are 1 to d, and new nodes continue with id from d + 1.
        start_contact : float
            Time the node started contact with node 0.
        end_contact : float
            Time the node stopped being a contact of node 0. Contacts at the end of the simulation
            have an end_contact equal to t_max.
    
    """

    # input validation:
    # d must be a positive integer
    if isinstance(d, bool) or not isinstance(d, (int, np.integer)):
        raise TypeError("Invalid input: d must be a positive integer.")
    if d < 1:
        raise ValueError(f"Invalid input: d must be a positive integer.")

    # lam must be a positive number
    if isinstance(lam, bool) or not isinstance(lam, (int, float)):
        raise TypeError(f"Invalid input: lam must be a positive number.")
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError(f"Invalid input: lam must be a positive number.")

    # t_max must be a positive number
    if isinstance(t_max, bool) or not isinstance(t_max, (int, float)):
        raise TypeError("Invalid input: t_max must be a positive number.")
    if not np.isfinite(t_max) or t_max <= 0:
        raise ValueError(f"Invalid input: t_max must be a positive number.")

    # simulation:
    # random number generator
    rng = np.random.default_rng(seed)

    # track and identify the number of contacts, constant d, of node 0
    slot_id = list(range(1, d + 1)) # slot_id[k] = id of node in slot k
    slot_start = [0.0] * d # slot_start[k] = time node starts contact with node 0

    # assign id to new nodes
    next_id = d + 1

    # initialize storage for finished contacts and their id, start contact time, and end contact time
    contacts = []

    # initalize simulation time
    t = 0.0

    # repeated simulation loop until t_max is reached
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
