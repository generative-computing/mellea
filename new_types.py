import random

class RangeType:
    """A custom range class that mimics the built-in range() behavior."""
    
    def __init__(self, start, stop=None, step=1):
        if step == 0:
            raise ValueError("RangeType() arg 3 must not be zero")

        if stop is None:
            self.start = 0
            self.stop = start
        else:
            self.start = start
            self.stop = stop

        self.step = step
        self.current = self.start

    def __iter__(self):
        """Returns the iterator object."""
        self.current = self.start  # Reset iterator for each new loop
        return self

    def __next__(self):
        """Returns the next value in the sequence."""
        if self.step > 0 and self.current >= self.stop:
            raise StopIteration
        if self.step < 0 and self.current <= self.stop:
            raise StopIteration

        value = self.current
        self.current += self.step
        return value

    def __repr__(self):
        """Official string representation of the object. Randomly selected age from a given interval."""
    
        return str(random.choice(list(self)))

