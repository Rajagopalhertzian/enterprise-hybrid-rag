from app.evaluation import recall_at_k, reciprocal_rank, ndcg_at_k

def test_recall():
    assert recall_at_k(["a","b"], {"a","c"}, 2) == 0.5

def test_mrr():
    assert reciprocal_rank(["x","b"], {"b"}) == 0.5

def test_ndcg():
    assert 0 < ndcg_at_k(["b","a"], {"a"}, 2) <= 1
    assert ndcg_at_k(["a"], {"a"}, 1) == 1.0
    assert ndcg_at_k(["x"], {"a"}, 1) == 0.0
