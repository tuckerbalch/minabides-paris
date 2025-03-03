from copy import deepcopy
import heapq as hq
from orders import Order, OrderBook
from random import random, randint, gauss
from statistics import mean
from util import ft, sign


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

    def message (self, ct, msg):
        """ Every agent must have this generator to yield back an arbitrary
            number of (rcp, msg) messages. """
        self.ct = ct
        if self.lob is None: yield (self.aid+1)%3, {}

    def new_day (self, start, evaluate=False):
        """ Called to receive information when starting a new simulated day (period).
            Evaluate is used to disable training in learning agent subclasses. """
        self.random_start(start)
        self.reset()

    def random_start (self, start):
        """ Randomizes exact agent start time with offset to prevent lock-step. """
        self.start = start + random() * self.offset

    def reset (self):
        """ Use to reset any information that should not carry over between
            simulated days. Called by new_day(). """
        pass


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

    def __init__ (self, symbol, minlat, interval, lot):
        super().__init__(symbol, minlat, interval, lot=lot, offset=1e9)
        self.done = False

    def message (self, ct, msg):
        super().message(ct, msg)
        if msg['type'] == 'lob':
            # Market maker agents cancel existing orders and place new ones near the mid.
            # Implementation option one: actually cancel and replace.  Slows simulation a bit.
            # Consider adding exchange "ladder" functionality to remove slowdown.
            for x in self.cancel_all(): yield x
            for i in range(4): yield self.place(self.lot, self.bid - i)
            for i in range(4): yield self.place(-self.lot, self.ask + i)

            # Implementation option two: Submit orders that expire about the time our next
            #                            orders should arrive.  Lower impact on simulation speed.
            #for i in range(4): yield self.place(self.lot, self.bid - i, exp=ct+1e6+2e3)
            #for i in range(4): yield self.place(-self.lot, self.ask + i, exp=ct+1e6+2e3)


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

    def __init__ (self, symbol, minlat, interval, lot):
        super().__init__(symbol, minlat, interval, lot=lot, offset=6e10)
        self.alpha, self.priv, self.surplus = 0.1, None, randint(5,100)

    def message (self, ct, msg):
        super().message(ct, msg)
        if msg['type'] == 'lob':
            fund = msg['fund']

            # Simple value agent updates belief, then arbs towards belief.
            # Consider changing scale of noise with price level.
            if self.priv is None: self.priv = int(fund + gauss()*100)
            else: self.priv += int(self.alpha * ((fund + gauss()*100) - self.priv))

            # Only trade if mid -> priv movement would be at least surplus.
            # Consider changing from mid to price-to-pay (best bid or ask).
            # Right now these are liquidity takers.  Could be providers.
            delta = self.priv - self.mid
            if abs(delta) < self.surplus: q = 0
            else: q = self.lot * sign(delta)

            for m in self.adjust(q, None): yield m


class ExchangeAgent(Agent):
    """ Represents the exchange.  Can handle multiple symbols each with its own order book.
        Understands placing market and limit orders, and reducing/canceling limit orders.
        Supports order expiration times (good until).  Maintains and delivers LOB snapshots
        and a configurable LOB snapshot history. """

    def __init__(self, symbols, args):

        super().__init__(0)
        self.args, self.symbols = args, symbols
        self.reset()

    def reset(self):
        self.book = { s: OrderBook(s, self.args) for s in self.symbols }
        self.notify = []

    def message(self, ct, msg):
        self.ct = ct
        mtype = msg['type']

        if mtype in ['exopen', 'exclose']: return

        # Handle subscription requests.
        if mtype == 'subscribe':
            hq.heappush(self.notify, (ct, msg['aid'], msg['levels'], msg['intvl']))
            return

        # All messages except exopen, exclose, subscribe contain an order.
        order = msg['order']

        # History is executed pre-market.  Nothing executes after hours.
        if (ct < self.args.exch_open and order.aid > 0) or ct > self.args.exch_close: return

        # Extract the relevant book for the current order.  Update its current time.
        book = self.book[order.symbol]
        book.ct = ct

        # Update fundamental if history order.
        if order.aid < 0 and 'fund' in msg: book.fund = msg['fund']

        # Handle the order
        if mtype == 'place':
            matching = True

            while matching:
                match = book.execute(order)
                if match:      # Order was at least partially matched.  Report out and continue.
                    filled = deepcopy(order)
                    filled.quantity = -match.quantity
                    filled.fill = match.fill
                    order.quantity -= filled.quantity

                    # Don't send execute messages to the history "agent".
                    if order.aid > 0: yield order.aid, { 'type': 'executed', 'order': filled }
                    if match.aid > 0: yield match.aid, { 'type': 'executed', 'order': match }

                    # Stop if order was completely exhausted.
                    if order.quantity == 0: matching = False
                else:
                    if order.limit is None:    # Market order couldn't be filled.
                        print (f"Warning: at {ft(ct)}, failed to fill market order {order}")
                        if order.aid > 0: yield order.aid, { 'type': 'rejected', 'order': order }
                    else:
                        book.enter(deepcopy(order))    # Add unfilled limit order to LOB.
                        if order.aid > 0: yield order.aid, { 'type': 'accepted', 'order': order }
                    matching = False

        elif mtype == 'cancel':
            cancelled = book.cancel(order)    # Cancellation could fail if executed already.
            result = 'cancelled' if cancelled else 'cancel_failed'
            if order.aid > 0: yield order.aid, { 'type': result, 'order': order }

        elif mtype == 'reduce':
            reduced = book.reduce(order)      # Reduction could fail if executed already.
            result = 'reduced' if reduced else 'reduce_failed'
            if order.aid > 0: yield order.aid, { 'type': result, 'order': order }

        # If market not open or no subscriptions, done.  Else send LOB subscriptions as needed.
        if ct < self.args.exch_open or len(self.notify) == 0: return

        # Build a snapshot to send to all agents in the subscription queue whose timer has
        # expired.  Each agent can request a different number of levels.
        if ct > self.notify[0][0]: book.clearexp()    # Clear expired orders before notifying.
        while ct > self.notify[0][0]:
            t, aid, lev, intvl = hq.heappop(self.notify)
            snap = book.snapview(lev)
            hq.heappush(self.notify, (ct + intvl, aid, lev, intvl))

            # Only send if we weren't processing this agent's own order.
            if aid != order.aid:
                yield aid, { 'type': 'lob', 'snap': snap, 'fund': book.fund, 'hist': book.snap_hist }

        # Uncomment this line to display an LOB snapshot after each inbound order is processed.
        #print(ft(ct), "\n" + book.summary(10))

