import numpy as np
import torch

from cam_deception.config import HammingConfig, ProjectedWTAConfig
from cam_deception.hamming import HammingWTAClassifier, binary_sign_np, hamming_similarity_np
from cam_deception.projected_lsq import WTAProjectedLSQClassifier, build_dot_product_preserving_projection
from cam_deception.splits import build_shared_example_split


def test_projection_preserves_dot_products_when_dim_covers_rank():
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(4, 16)).astype(np.float32)
    W, meta = build_dot_product_preserving_projection(rows, d_proj=8, seed=0)
    x = rng.normal(size=(7, 16)).astype(np.float32)
    rows_norm = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    before = x @ rows_norm.T
    after = (x @ W.T) @ (rows_norm @ W.T).T
    assert meta["exact_when_projection_dim_covers_rank"]
    np.testing.assert_allclose(after, before, atol=2e-5, rtol=2e-5)


def test_euclidean_and_dot_wta_scores_match_without_quantization():
    rng = np.random.default_rng(1)
    rows = rng.normal(size=(5, 12)).astype(np.float32)
    x = torch.tensor(rng.normal(size=(9, 12)).astype(np.float32))
    cfg = ProjectedWTAConfig(use_projection=True, projection_dim=8, use_lsq_qat=False)
    model = WTAProjectedLSQClassifier(rows, cfg).eval()
    with torch.no_grad():
        dot_scores, dot_winners, _ = model.forward_dot(x, quantize=False)
        eu_scores, eu_winners, _ = model.forward_euclidean(x, quantize=False)
    torch.testing.assert_close(eu_scores, dot_scores, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(eu_winners, dot_winners)


def test_unweighted_hamming_is_raw_max_similarity_plus_bias():
    rows = np.array([[1, -1, 1, -1], [-1, -1, 1, 1]], dtype=np.float32)
    x = np.array([[2, -3, 4, -1], [-2, -4, 2, 3]], dtype=np.float32)
    model = HammingWTAClassifier(rows, method="sign", weighted_rows=False).eval()
    with torch.no_grad():
        scores, winners, _ = model(torch.tensor(x))
    expected = hamming_similarity_np(binary_sign_np(x), binary_sign_np(rows))
    np.testing.assert_allclose(scores.numpy(), expected.max(axis=1), atol=1e-6)
    np.testing.assert_array_equal(winners.numpy(), expected.argmax(axis=1))
    np.testing.assert_allclose(model.alphas.detach().numpy(), np.ones(2), atol=1e-6)


def test_shared_split_has_no_example_leakage():
    n = 80
    domains = np.array(["a", "b"] * (n // 2))
    labels = np.array(([0, 1, 0, 1] * (n // 4)))
    acts = {
        "example_index_mean": np.arange(n),
        "domain_mean": domains,
        "y_mean": labels,
    }
    split = build_shared_example_split(acts, random_seed=0)
    train, val, test = map(lambda k: set(split[k].tolist()), ["train", "val", "test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(range(n))
