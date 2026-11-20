from app.evaluation import recall_at_k, reciprocal_rank, ndcg_at_k

def test_recall():
    assert recall_at_k(["a","b"], {"a","c"}, 2) == 0.5

def test_mrr():
    assert reciprocal_rank(["x","b"], {"b"}) == 0.5

def test_ndcg():
    assert 0 < ndcg_at_k(["b","a"], {"a"}) <= 1
