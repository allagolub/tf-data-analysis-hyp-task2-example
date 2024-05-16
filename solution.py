import pandas as pd
import numpy as np
from hyppo.ksample import MMD


chat_id = 1304567965

def solution(x: np.array, y: np.array) -> bool:
    res = MMD(compute_kernel='rbf', gamma=1.0).test(x, y)
    return res[1] < 0.06
