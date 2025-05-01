
from collections import namedtuple
import concurrent.futures, numpy as np, os, random, sys, torch
from datetime import datetime as dt


### Utility classes.

class Box():
    """ A very simple continuous space in the style of the gymnasium Box.
        Does not support unbounded spaces. """
    def __init__(self, low, high):
        self.low, self.high = low, high
        self.shape = self.low.shape

    def sample(self):
        """ Uniformly randomly samples from this bounded space. """
        return np.random.uniform(low=self.low, high=self.high, size=self.shape)

class Discrete():
    """ A very simple space in the style of the gynmasium Discrete. """

    def __init__(self, n, start = 0):
        self.n, self.start = n, start       # length and starting number of action space range
        self.shape = (1,)

    def sample(self):
        """ Uniformly randomly samples from this space. """
        return np.random.randint(low=self.start, high=self.start+self.n)

# Simple named tuple type for replay samples.
ReplaySample = namedtuple('ReplaySample', ['observations', 'actions', 'next_observations', 'rewards'])

class ReplayBuffer():
    """ A very simple replay buffer in the style of stable-baselines3. """
    def __init__(self, maxlen, obs_shape, act_shape, discrete_actions = False):
        self.s = np.empty((maxlen, *obs_shape), dtype=np.float32)
        self.a = np.empty((maxlen, *act_shape), dtype=np.int64 if discrete_actions else np.float32)
        self.s_prime = np.empty((maxlen, *obs_shape), dtype=np.float32)
        self.r = np.empty((maxlen), dtype=np.float32)

        self.n = 0
        self.full = False

    def add(self, s, a, s_prime, r):
        self.s[self.n] = np.array(s)
        self.a[self.n] = np.array(a)
        self.s_prime[self.n] = np.array(s_prime)
        self.r[self.n] = np.array(r)

        self.n = self.n + 1 % self.s.shape[0]
        if self.n == 0: self.full = True

    def sample(self, num_samples):
        idx = np.random.randint(0, self.s.shape[0] if self.full else self.n, size=num_samples)
        return ReplaySample(torch.tensor(self.s[idx]), torch.tensor(self.a[idx]),
                            torch.tensor(self.s_prime[idx]), torch.tensor(self.r[idx]))

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


### Utility methods.

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


