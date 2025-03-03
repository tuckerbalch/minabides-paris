import concurrent.futures, numpy as np, os, random, sys, torch
from datetime import datetime as dt

def clamp(n, smallest, largest):
    """ Clamp numeric value n to be within inclusive range [smallest,largest]. """
    return max(smallest, min(n, largest))

def ft(t):
    """ Use sparingly to format readable time.  Slow! """
    return dt.fromtimestamp(ft.date_ts + t / 1e9)

def latency(minlat):
    """ Sample latency for a message given minimum latency for agent. """
    jitter = 0.5
    clip = 0.1
    unit = 10.0

    x = random.uniform( clip, 1.0 )

    # Note: with some HFT agents, allowing jitter can create timing issues, as the agent may be
    #       triggered by LOB messages to take new actions before all previous execution messages
    #       are received.  In this case, it may be helpful to simply return minlat.  This allows
    #       each agent its own latency, but keeps it constant per agent.
    #
    #       This issue has been substantially addressed by having the exchange not send LOB updates
    #       to the agent whose incoming order triggered the update.

    return minlat + (jitter / x**3) * (minlat / unit)

def set_manual_seed (seed):
    """ This is probably overkill, but set every possible value that can affect determinism. """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sign (n):
    """ Return the sign of the numeric parameter. """
    if n == 0: return 0
    elif n > 0: return 1
    else: return -1


class TeeOutput:
    """ Used by main simulation to automatically tee output to a file. """
    def __init__(self, tee_path):
        self.screen = sys.stdout
        self.file = open(tee_path, "wb", buffering=0)

    def write(self, text):
        self.screen.write(text)
        self.file.write(text.encode('utf-8'))

    def flush(self):
        """ For filestream compatibility. """
        pass

