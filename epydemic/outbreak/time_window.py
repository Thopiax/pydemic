import pandas as pd

from typing import Optional, Union

from outbreak.exceptions import InvalidWindowBurnInException, InvalidWindowStartException, InvalidWindowEndException


class OutbreakTimeWindow:
    def __init__(self, outbreak, start: int = 0, end: Optional[Union[float, int]] = None,
                 burn_in_minimum_deaths_threshold: int = 10):

        self.outbreak = outbreak

        if burn_in_minimum_deaths_threshold > 0:
            self.burn_in_time = outbreak.ffx(burn_in_minimum_deaths_threshold, x_type="deaths")

            if self.burn_in_time is None:
                raise InvalidWindowBurnInException

            # start counts on top of burn-in
            start += self.burn_in_time

        self.start = start

        if self.start >= len(outbreak):
            raise InvalidWindowStartException

        if end is None:
            self.end = len(outbreak)
        elif 0 < end < 1:
            # end is the proportion of the outbreak to be considered since start
            self.end = self.start + int((len(outbreak) - self.start) * end)
        elif end <= len(outbreak):
            self.end = end
        else:
            raise InvalidWindowEndException

        # ensure time-window is well defined
        self.end = max(self.end, self.start)

    def __len__(self):
        return self.end - self.start

    # return any outbreak Series attribute (e.g. cases) within the window frame.
    def __getattr__(self, item):
        if hasattr(self.outbreak, item):
            attr = getattr(self.outbreak, item)

            if type(attr) is pd.Series:
                return attr.iloc[self.start:self.end]

            return attr

        return None

