from typing import Tuple
from operator import itemgetter


def get_multiple_vals_from_dict(cls, *args, obj: dict) -> Tuple:
    res = itemgetter(*args)(obj)
    if len(res) == 1:
        return (res,)
    return res
