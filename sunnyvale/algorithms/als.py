from datetime import datetime
import itertools
import os

import numpy as np
import scipy.sparse as sparse

import als_cython


class ALS:
    def __init__(self, factors=100, alpha=1, lambda_param=0.01, iterations=15):
        self._factors = factors
        self._alpha = alpha
        self._lambda = lambda_param
        self._iterations = iterations

        self._n_users = 0
        self._n_items = 0

        self._X = None
        self._Y = None
        # csr_matrix type
        self._R = None


    def fit(self, R, random_seed=None):
        print('Sart ALS fit...')
        total_t1 = datetime.now()

        self._R = R
        self._n_users, self._n_items = R.shape

        # X : user-factors within an m × f matrix
        # Y : item-factors within an n × f matrix
        np.random.seed(random_seed)
        self._X = np.random.randn(self._n_users, self._factors).astype(np.float32)
        np.random.seed(random_seed)
        self._Y = np.random.randn(self._n_items, self._factors).astype(np.float32)

        # C = 1 + self._alpha * R (c_ui = 1 + alpha * r_ui)
        alphaR = self._alpha * R

        # (lambda * I)는 상수
        lambdaI = self._lambda * np.eye(self._factors).astype(np.float32)

        for iteration in range(self._iterations):
            t1 = datetime.now()

            self._update_user_vectors(alphaR, self._X, self._Y, lambdaI)
            self._update_item_vectors(alphaR, self._X, self._Y, lambdaI)

            t2 = datetime.now()
            print('iteration: {}/{}, runnig time: {}'.format(iteration+1, self._iterations, t2-t1))

        total_t2 = datetime.now()
        print('Total runnig time: {}'.format(total_t2 - total_t1))


    def similar_items(self, item_idx, topk=10):
        y_i = self._Y[item_idx]

        dot_product = np.dot(self._Y, y_i)
        l2_norm = np.linalg.norm(self._Y, axis=1) * np.linalg.norm(y_i)

        similarity = dot_product / l2_norm

        best = np.argpartition(similarity, -topk)[-topk:]
        return sorted(zip(best, similarity[best]), key=lambda x: -x[1])


    def recommend(self, user_idx, topk=10):
        x_u = self._X[user_idx]

        scores = np.dot(self._Y, x_u)

        filter_items = set()
        filter_items.update(self._R[user_idx].indices)

        count = topk + len(filter_items)

        best = np.argpartition(scores, -count)[-count:]
        best_items = sorted(zip(best, scores[best]), key=lambda x: -x[1])

        return list(itertools.islice((item for item in best_items if item[0] not in filter_items), topk))


    # 아래 single thread 버전은 성능 문제로 중간에 drop....
    # x_u = (YtCuY + lambda*I)^-1 (Yt*Cu*p(u))
    #     = (YtY + Yt(Cu-I)Y + lambda*I)^-1 (Yt*Cu*p(u))
    #     = (c_u*YtY + lambda*I)^-1 (c_u*Yt*p(u))
    #     = (c_u*YtY + lambda*I)^-1 (Yt(p(u) + alpha*r_u))
    def _update_user_vectors(self, alphaR, X, Y, lambdaI):
        YtY = Y.T.dot(Y)

        for u in range(self._n_users):
            r_u = alphaR[u, :].toarray().flatten()

            # c_u = 1 + alpha * r_u
            c_u = 1 + r_u.copy()
            # preference p(u)
            p_u = r_u.copy()
            p_u[np.where(p_u != 0)] = 1.0

            # Cu - I, Cu (c_ui = 1 + alpha * r_ui)
            # Cu_I = np.diag(r_u)
            # Cu = Cu_I + np.eye(self._n_items)

            # A = (c_u*YtY + lambda*I)
            # b = (Yt(p(u) + alpha*r_u))
            A = np.dot(c_u * Y.T, Y) + lambdaI
            b = np.dot(Y.T, (p_u + r_u))

            # A * x_u = b
            X[u] = np.linalg.solve(A, b)


    # y_i = (XtCiX + lambda*I)^-1 (Xt*Ci*p(i))
    #     = (XtX + Xt(Ci-I)X + lambda*I)^-1 (Xt*Ci*p(i))
    #     = (c_i*XtX + lambda*I)^-1 (c_i*Xt*p(i))
    #     = (c_i*XtX + lambda*I)^-1 (Xt(p(i) + alpha*r_i))
    def _update_item_vectors(self, alphaR, X, Y, lambdaI):
        XtX = X.T.dot(X)

        for i in range(self._n_items):
            r_i = alphaR[:, i].T.toarray().flatten()

            # c_i = 1 + alpha * r_i
            c_i = 1 + r_i.copy()
            # preference p(i)
            p_i = r_i.copy()
            p_i[np.where(p_i != 0)] = 1.0

            # Ci - I, Ci
            # Ci_I = np.diag(r_i)
            # Ci = Ci_I + np.eye(self._n_users)

            # A = (c_i*XtX + lambda*I)
            # b = (Xt(p(i) + alpha*r_i))
            A = np.dot(c_i * X.T, X) + lambdaI
            b = np.dot(X.T, (p_i + r_i))

            # A * y_i = b
            Y[i] = np.linalg.solve(A, b)
