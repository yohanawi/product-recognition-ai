import unittest

import numpy as np

import vector_search


class VectorSearchTests(unittest.TestCase):
    def test_normalize_rows_returns_unit_vectors(self):
        vectors = vector_search.normalize_rows([[3, 4], [5, 12]])
        norms = np.linalg.norm(vectors, axis=1)
        self.assertTrue(np.allclose(norms, np.ones_like(norms)))

    def test_cosine_similarity_prefers_closest_candidate(self):
        scores = vector_search.cosine_similarity(
            [1, 0, 0],
            [
                [1, 0, 0],
                [0.6, 0.8, 0],
                [0, 1, 0],
            ],
        )

        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])

    def test_build_faiss_index_requires_dependency(self):
        if vector_search.faiss is not None:
            self.skipTest("faiss is installed; dependency guard is not applicable")

        with self.assertRaises(vector_search.FaissUnavailableError):
            vector_search.build_faiss_index([[1, 0, 0]])


if __name__ == "__main__":
    unittest.main()
