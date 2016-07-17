#encoding=utf-8

import pandas as pd
import numpy as np
import time

if __name__ == '__main__':
    vec_1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    vec_2 = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,]

    vec_1 = np.array(vec_1)
    vec_2 = np.array(vec_2)

    time_1 = time.time()

    print np.sqrt(np.sum(np.square(vec_1 - vec_2)))

    time_2 = time.time()
    print time_2-time_1

    print np.linalg.norm(vec_1 - vec_2)

    time_3 = time.time()
    print time_3-time_2
