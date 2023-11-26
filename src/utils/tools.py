from scipy.optimize import brute, minimize
from tqdm import tqdm
import numpy as np


def accept_reject(f, xlim, samples, hide_pbar=False):

    # find maximum of f
    argmax = brute(lambda x: -f(x), [xlim], disp=False)[0]
    argmax = minimize(lambda x: -f(x), argmax).x[0]
    fmax = f(argmax)

    events = []

    with tqdm(total=samples, disable=hide_pbar) as pbar:
        while len(events) < samples:
            # generate a random value along xrange = x
            x = np.random.uniform(*xlim)
            # evaluate function at x
            val = f(x)

            if val < 0:
                raise RuntimeError(f"Found negative value for pdf: f({x=}) = {val}")

            # TODO: handle fmax violations better
            if val > fmax:
                print(f"Found value above fmax: f({x}) = {val} > {fmax}, updating fmax...")
                fmax = val

            # generate random number between 0 and fmax = y
            y = np.random.uniform(0, fmax)
            # if y <= f(x)  then accept x
            if y <= val:
                events.append(x)
                pbar.update(1)

    return events

