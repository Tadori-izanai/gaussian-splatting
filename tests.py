#
# Created by lxl.
#

import numpy as np
from utils.general_utils import knn
from arguments import GroupParams

def test_knn():
    points = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 3], [3, 3, 4], [4, 4, 2]])
    dist, idx = knn(points, 2)
    for i in np.arange(len(points)):
        print(points[i], '\n', idx, '\n', dist, '\n')
        print()
    print(points[idx])

if __name__ == '__main__':
    test_knn()
