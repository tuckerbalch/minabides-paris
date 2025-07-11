import csv, glob, os, pickle
from orders import Order, OrderBook, OrderLevel
from util import ft

class History:
    """ This class is responsible for reading LOBSTER-style HFT data files,
        constructing an order stream, and delivering it in requested batches
        to the simulation kernel. It can also cache a starting point to
        permit future fast-forwarding to a previously used start time. """

    def __init__ (self, symbol, date, start, replay, fund_type, data_dir):

        self.symbol = symbol
        self.date = date
        self.start = start
        self.replay = replay
        self.fund_type = fund_type
        self.fund = None

        year = date[:4]
        month = date[5:7]

        data_dir = os.path.expanduser(data_dir)
        self.data_dir = data_dir

        # Open file, read first record, keep as next.  File stays open.  Glob should match one file.
        fileglob = f"{symbol}_{date}_*_message_0.csv"
        files = glob.glob(os.path.join(data_dir, fileglob))
        if len(files) == 0:
            print (f"Could not find requested data file: {os.path.join(data_dir, fileglob)}")
            exit()
        self.file = open(files[0], newline="")
        self.data = csv.reader(self.file)

        self.next = self.get_next()

    def get_next (self):
        """ Read and process the single next order message. """
        while True:                     # python has no do-while
            n = next(self.data, None)   # get next message
            if n is None: return n      # stop if at end of file
            n = n[:6]                   # LOBSTER bug currently includes MPID even if requested not to
            n[0] = float(n[0]) * 1e9    # to ns
            n = [int(x) for x in n]     # everything to int
            n[4] //= 100                # to cents
            if n[1] not in [1,2,3,4,5]: continue    # skip unwanted message types
            break
        return n

    def history (self, et):
        """ Generator that will yield (delivery time, msg) orders for the exchange.
            et is end time for the current history queueing. """

        while True:
            if self.next is None: break
            n = self.next

            # Stop if about to pass requested end time.
            if n[0] >= et: break

            # During history replay, fundamental is simply the most recently
            # executed real-world trade (not simulated market).
            # Could change to bayesian VWAP of executed trades or similar.
            # Fixed fundamental retains the value of the first executed trade.
            if n[1] in (4,5):
                if self.fund_type == 'history': self.fund = n[4]
                elif self.fund_type == 'fixed' and self.fund is None: self.fund = n[4]

            if self.replay:
                otype = 'place'
                if n[1] == 2: otype = 'reduce'
                elif n[1] == 3: otype = 'cancel'
            else:
                otype = 'fund'

            # Execute orders are given a new oid, other types pass through oid.
            o = Order(-1, n[3] if n[5] > 0 else -n[3], self.symbol, n[4], oid = None if n[1] in (4,5) else n[2])

            # Execute orders have quantity negated (e.g. buy/sell) and are passed as 'place'.
            if n[1] in (4,5): o.quantity *= -1

            #if n[1] == 5: otype = 'ignore'    # needed if trying to match lobster LOB

            # Yield the next requested historical order and continue until requested end time.
            yield n[0], { 'type': otype, 'fund': self.fund, 'order': o }
            self.next = self.get_next()

    def fast_forward (self, et):
        """ Fast forwards data file to requested end time. """

        while (True):
            if self.next is None: return
            if self.next[0] > et: return

            self.next = self.get_next()

    def reconstruct (self, et, cachefile, args):
        """ Reconstructs LOB to requested end time and saves it for future fast-forwarding. """
        book = OrderBook(self.symbol, args)
    
        while True:
            if self.next is None: break
            n = self.next
            if n[0] >= et: break
    
            # Visible or hidden execution becomes the new fundamental, unless fixed.
            if n[1] in (4,5):
                if self.fund_type == 'history': book.fund = n[4]
                elif self.fund_type == 'fixed' and book.fund is None: book.fund = n[4]

            o = Order(-1, n[3] if n[5] > 0 else -n[3], self.symbol, n[4], oid = n[2])
    
            if n[1] in [1,2,3,4]: book.ct = n[0]    # update book's last order time

            if n[1] == 1:   book.enter(o)
            elif n[1] == 2: book.reduce(o)
            elif n[1] == 3: book.cancel(o)
            elif n[1] == 4: book.reduce(o)
    
            self.next = self.get_next()

        with open(cachefile, 'wb') as out: pickle.dump(book, out, pickle.HIGHEST_PROTOCOL)
    
    def close (self):
        """ Closes the history file.  Call at end of each day. """
        self.file.close()

