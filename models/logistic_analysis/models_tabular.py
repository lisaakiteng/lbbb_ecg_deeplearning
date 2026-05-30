from sklearn.linear_model import LogisticRegression

def build_logistic_basic():
    return LogisticRegression(max_iter=5000, class_weight="balanced")

def build_logistic_full():
    return LogisticRegression(max_iter=5000, class_weight="balanced")