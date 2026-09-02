import numpy as np
from src.directions import mean_difference_direction, projection_scores, auc


def test_mean_difference_direction_separates_simple_data():
    benign = np.array([[-2.0, 0.0], [-1.0, 0.1], [-1.5, -0.1]])
    harmful = np.array([[1.0, 0.0], [2.0, 0.1], [1.5, -0.1]])
    x = np.vstack([benign, harmful])
    y = np.array([0, 0, 0, 1, 1, 1])
    d, center, gap = mean_difference_direction(x, y)
    scores = projection_scores(x, d)
    assert np.isclose(np.linalg.norm(d), 1.0)
    assert gap > 0
    assert auc(y, scores) == 1.0
    assert -0.5 < center < 0.5
