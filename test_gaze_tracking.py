import unittest
import numpy as np
from gaze_tracking import get_iris_center, solve_mapping, main


class DummyLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class TestGazeMouseFunctions(unittest.TestCase):
    def test_get_iris_center_uniform(self):
        # Create landmarks all at (0.5, 0.5)
        landmarks = [DummyLandmark(0.5, 0.5) for _ in range(4)]
        img_w, img_h = 640, 480
        # All landmarks map to the center
        cx, cy = get_iris_center(landmarks, list(range(4)), img_w, img_h)
        expected_x = 0.5 * img_w
        expected_y = 0.5 * img_h
        self.assertAlmostEqual(cx, expected_x)
        self.assertAlmostEqual(cy, expected_y)

    def test_get_iris_center_varying(self):
        # Landmarks at different corners
        landmarks = [
            DummyLandmark(0.0, 0.0),
            DummyLandmark(1.0, 0.0),
            DummyLandmark(1.0, 1.0),
            DummyLandmark(0.0, 1.0),
        ]
        img_w, img_h = 800, 600
        # Average x = 0.5, average y = 0.5
        cx, cy = get_iris_center(landmarks, list(range(4)), img_w, img_h)
        self.assertAlmostEqual(cx, 0.5 * img_w)
        self.assertAlmostEqual(cy, 0.5 * img_h)

    def test_solve_mapping_identity(self):
        # Simple identity mapping: src equals dst
        src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        dst = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        M = solve_mapping(src, dst)
        # M should map [u, v, 1] -> [u, v]
        # Expect M = [[1,0], [0,1], [0,0]]
        expected = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        np.testing.assert_allclose(M, expected, atol=1e-6)

    def test_solve_mapping_affine(self):
        # Test mapping to an offset and scale: x = 100*u + 100, y = 100*v + 200
        src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        dst = np.array([[100.0, 200.0], [200.0, 200.0], [100.0, 300.0]])
        M = solve_mapping(src, dst)
        expected = np.array([[100.0, 0.0], [0.0, 100.0], [100.0, 200.0]])
        np.testing.assert_allclose(M, expected, atol=1e-6)

    def test_main(self):
        main()

if __name__ == "__main__":
    unittest.main()
