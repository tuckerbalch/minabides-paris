""" This module contains the basic agent and trading agent classes,
    one of which should be the parent class for all simulation participants.
"""

from orders import Order
from random import random
from util import ft


class Agent:
    """ Basic simulation agent with no special abilities. """
    next_aid = 0        # shared class variable for autoincrements

    def __init__ (self, minlat, offset=0.0, tag=''):
        self.aid = Agent.next_aid
        Agent.next_aid += 1

        self.minlat, self.offset, self.tag = minlat, offset, tag
        self.type = self.__class__.__name__
        self.name = f"{self.type}{self.tag} {self.aid}"
        self.ct = None
        self.stagger = random()

    def message (self, ct, msg):
        """ Every agent must have this generator to yield back an arbitrary
            number of (rcp, msg) messages. """
        self.ct = ct
        if self.lob is None: yield (self.aid+1)%3, {}

    def new_day (self, start, evaluate=False, fixed=False):
        """ Called to receive information when starting a new simulated day (period).
            Evaluate is used to disable training in learning agent subclasses. """
        self.eval = evaluate
        if not fixed: self.stagger = random()
        self.random_start(start)
        self.reset()

    def random_start (self, start):
        """ Randomizes exact agent start time with offset to prevent lock-step. """
        self.start = start + self.stagger * self.offset

    def reset (self):
        """ Use to reset any information that should not carry over between
            simulated days. Called by new_day(). """
        pass

    def report_loss(self):
        """ For learning agents, should return the current episode, episode step, global step, actor loss,
            and critic loss for logging (and eventually plotting).  Other agents return nothing. """
        return



class TradingAgent(Agent):
    """ The base trading agent.  Comes with a lot of built in functionality. """

    def __init__ (self, symbol, minlat, interval, lot=None, offset=0.0, tag=''):
        super().__init__(minlat, offset=offset, tag=tag)
        self.interval, self.symbol, self.lot = interval, symbol, lot
        self.exch, self.held, self.open, self.cash, self.portval = [0] * 5
        self.snap, self.bid, self.ask, self.bidq, self.askq, self.mid, self.start_pv = [None] * 7
        self.last_exec = None
        self.orders = {}
        self.lob_req = []           # Agents can request extra info in LOB messages.

    def _debug(self, m):
        """ Allows turning on/off debug printing by adding debug attribute to agent. """
        if hasattr(self, 'debug'): print(f"{ft(self.ct)} {self.name}: {m}")

    def adjust(self, tgt, p=None, tag=None):
        """ Places or cancels orders to adjust holdings to target with optional limit price. """
        cur = self.open + self.held

        if cur != tgt and self.open != 0:
            # Might cancel some.
            canc = self.orders.copy()
            for oid,o in canc.items():
                if (cur > tgt and o.quantity > 0) or (cur < tgt and o.quantity < 0):
                    cur -= o.quantity
                    yield self.cancel(oid)

        q = tgt - cur
        if q != 0: yield self.place(q,p,tag=tag)

    def cancel(self, oid):
        """ Generates a message tuple to yield for cancelling an order by id. """
        self._debug(f"cancelling {self.orders[oid]}")
        return self.exch, { 'type': 'cancel', 'order': self.orders[oid] }

    def cancel_all(self):
        """ Yields a series of message tuples to cancel all outstanding orders. """
        self._debug(f"cancelling all")
        for oid,o in self.orders.copy().items():
            yield self.exch, { 'type': 'cancel', 'order': o }

    def cost(self, cost):
        """ Called to apply a transaction cost to an agent's portfolio.
            Separated to allow fine control in experiments. """
        self.cash -= cost

    def execute(self, q, p):
        """ Called when exchange confirms order execution.  Does bookkeeping. """
        self.held += q
        self.cash -= q * p
        self._debug(f"executed: {q},{p}")
        self.last_exec = q,p
        self.mark()

    def finalize_episode(self, ct, msg):
        """ Handle the final update of the episode.  Expects an lob message
            to mark to market.  Returns episode profit. """
        self.ct = ct
        self.handle(ct, msg)

        profit = self.portval / 100
        return profit

    def handle(self, ct, msg):
        """ Top level method to handle a variety of incoming messages. """
        mt = msg['type']
        if mt == 'lob': self.track(msg['snap'])
        else:
            o = msg['order']
            if mt == 'executed': self.execute(o.quantity, o.fill)
            if mt in ['cancelled','executed','reduced','rejected']:
                if o.oid in self.orders:
                    oo = self.orders[o.oid]
                    self.open -= o.quantity
                    oo.quantity -= o.quantity
                    if mt in ['cancelled','rejected'] or oo.quantity == 0: del self.orders[o.oid]

    def mark(self):
        """ Marks current portfolio to market using internal information. """
        self._debug(f"portval: {self.held} * {self.mid} + {self.cash} == {self.held*self.mid+self.cash}")
        self.portval = self.held * self.mid + self.cash  # could use bid or ask depending on holdings, but maybe not

    def message(self, ct, msg):
        """ Common message processing behavior for all trading agents. """
        self.ct = ct
        self.handle(ct, msg)

    def place(self, q, p=None, exp=None, tag=None, track=True, info=None):
        """ Returns an order message tuple (to yield) for placing a market or limit order. """
        self._debug(f"bid {self.bidq}@{self.bid}, ask {self.askq}@{self.ask}, placed: {q},{p}")

        # add to open orders when placed, remove if rejected
        o = Order(self.aid, q, self.symbol, p, tag=tag, exp=exp)
        if track: self.open, self.orders[o.oid] = self.open + o.quantity, o
        if info is not None: return self.exch, { 'type': 'place', 'order': o, 'info' : info }
        else: return self.exch, { 'type': 'place', 'order': o }

    def summarize(self, levels=100):
        """ Returns a string summary of the LOB from this agent's perspective. """
        levels = min(levels, max(len(self.snap['bid']), len(self.snap['ask'])))
        s = f"{'BID':>6} {'PRICE':>8} {'ASK':>6}\n{'-'*3:>6} {'-'*5:>8} {'-'*3:>6}\n"
        for a in reversed(self.snap['ask'][:levels]): s += f"{'':6} {a.price/100:8.2f} {a.quantity:6d}\n"
        for b in self.snap['bid'][:levels]: s += f"{b.quantity:6d} {b.price/100:8.2f}\n"
        return s

    def track(self, snap):
        """ Stores an LOB snapshot, computes other attributes, and marks to market. """
        self.snap = snap
        self.ask, self.bid = snap['ask'][0].price, snap['bid'][0].price
        self.askq, self.bidq = snap['ask'][0].quantity, snap['bid'][0].quantity
        self.mid = (self.ask + self.bid) / 2
        self.mark()
        if self.start_pv is None: self.start_pv = self.portval

