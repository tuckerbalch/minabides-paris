""" This module contains a number of very simple trading agents which can
    be used as the "background" population of a market.
"""

from market.agent import TradingAgent
from math import ceil, floor
from random import random, randint, gauss
from statistics import mean
from util import ft, sign


class MomentumAgent(TradingAgent):
    """ Simple momentum agent that tries to ride the trend. """

    def __init__ (self, symbol, minlat, interval, lot):
        super().__init__(symbol, minlat, interval, lot=lot, offset=6e10)
        self.n_pm, self.prevmids = 5, []

    def message (self, ct, msg):
        super().message(ct, msg)
        if msg['type'] == 'lob':
            # Momentum agents place market orders based on recent price trend.
            if len(self.prevmids) >= self.n_pm:
                delta = self.mid - mean(self.prevmids)
                if abs(delta) < 5: delta = 0
                q = self.lot * sign(delta)
                for m in self.adjust(q): yield m

            self.prevmids = ([self.mid] + self.prevmids)[:self.n_pm]


class MarketMakerAgent(TradingAgent):
    """ Simple market maker that places a ladder around the mid-price. """

    def __init__ (self, symbol, minlat, interval, lot, spread=1, strategy='expire'):
        super().__init__(symbol, minlat, interval, lot=lot, offset=1e9)
        self.spread, self.strategy, self.done = spread, strategy, False

    def message (self, ct, msg):
        super().message(ct, msg)
        if msg['type'] == 'lob':
            # Market maker agents cancel existing orders and place new ones near the mid.
            # Note: maintains a one unit spread.  A smarter agent would vary this.
            # Implementation option one: actually cancel and replace.  Slows simulation a bit.
            if self.strategy == 'cancel':
                to_cancel = self.cancel_all()
                for i in range(4): yield self.place(self.lot, ceil(self.mid)-i-self.spread)
                for i in range(4): yield self.place(-self.lot, floor(self.mid)+i+self.spread)
                for x in to_cancel: yield x

            # Implementation option two: Submit orders that expire about the time our next
            #                            orders should arrive.  Lower impact on simulation speed.
            elif self.strategy == 'expire':
                for i in range(4): yield self.place(self.lot, ceil(self.mid)-i-self.spread, exp=ct+1.1*self.interval)
                for i in range(4): yield self.place(-self.lot, floor(self.mid)+i+self.spread, exp=ct+1.1*self.interval)


class NoiseAgent(TradingAgent):
    """ Simple noise agent that places random orders. """

    # To improve speed, the noise agent could place orders ahead as the history agent does.
    # It could also act as any number of uninformed participants.

    def __init__ (self, symbol, minlat, interval):
        super().__init__(symbol, minlat, interval, offset=6e10)
        self.done = False

    def message (self, ct, msg):
        super().message(ct, msg)
        if msg['type'] == 'lob':
            # Noise agents don't see fund.  They place random orders near the current mid.
            q = randint(50,150) * (1 if random() < 0.5 else -1)
            p = int(self.mid + gauss() * 100)
            yield self.place(q,p)


class OrderBookImbalanceAgent(TradingAgent):
    """ A low-latency trading agent that follows an order book imbalance indicator. """

    def __init__ (self, symbol, minlat, interval, lot, levels, tag=''):
        super().__init__(symbol, minlat, interval, lot=lot, tag=tag, offset=1e9)
        self.levels = levels

    def message (self, ct, msg):
        super().message(ct, msg)
        if msg['type'] == 'lob':
            # OBI agents place orders based on volume imbalance weighted by
            # distance from the mid-price.
            b, a, fac = 0, 0, 1.0
            for i in range(min(len(self.snap['bid']), len(self.snap['ask']))):
                if i != 0 and i % 5 == 0: fac *= 2
                b += self.snap['bid'][i].quantity / fac
                a += self.snap['ask'][i].quantity / fac
            imba = b / (b+a)

            # Current shares held or in unfilled orders.
            cur = self.held + self.open

            # Thresholds for opening and closing positions.  Open long, open short, close.
            olth, osth, cth = 0.6, 0.4, 0.5

            # Imbalance logic differs with current holdings.
            if cur == 0:
                if imba > osth and imba < olth: q = 0             # flat
                else: q = self.lot * (1 if imba >= olth else -1)  # open
            elif cur > 0:
                if imba > cth: q = self.lot                       # keep
                elif imba > osth: q = 0                           # flat
                else: q = -self.lot                               # flip
            else:
                if imba < cth: q = -self.lot                      # keep
                elif imba < olth: q = 0                           # flat
                else: q = self.lot                                # flip

            self._debug(f"{ft(ct)}: held: {self.held}, mid {self.mid}, imba {imba:.2}")

            # Only call adjust if changes need to be made.
            if cur != q:
                buy = q > 0
                for m in self.adjust(q, p = self.ask+10 if buy else self.bid-10): yield m


class ValueAgent(TradingAgent):
    """ Simple value agent that arbitrages the market towards the fundamental. """

    def __init__ (self, symbol, minlat, interval, lot, noise=5, alpha=1.0):
        super().__init__(symbol, minlat, interval, lot=lot, offset=6e10)
        self.alpha, self.noise, self.priv = alpha, noise, None

    def message (self, ct, msg):
        super().message(ct, msg)
        if msg['type'] == 'lob':
            fund = msg['fund']

            # Simple value agent updates belief, then arbs towards belief.
            # Consider changing scale of noise with price level.
            if self.priv is None: self.priv = int(fund + gauss()*self.noise)
            else: self.priv += int(self.alpha * ((fund + gauss()*self.noise) - self.priv))

            delta = self.priv - self.mid
            if delta == 0: return    # mid-price equals private valuation
            else: q = self.lot * sign(delta)

            # Chance: 1/3 join top of book, 1/3 limit at mid, 1/3 take market price.
            price = None
            orand = random()
            if orand < 0.333:
                if q < 0: price = self.ask              # join best ask
                else:     price = self.bid              # join best bid
            elif orand < 0.666: price = int(self.mid)   # limit at mid-price
            else: pass                                  # take market price

            # Cancel old orders and place new.
            for x in self.cancel_all(): yield x
            yield self.place(q, price)

