"""Boost tests for sparse.py and statistics.py."""

from tests.conftest import run_matlab, get_val
from matpy.runtime.types import Mat


class TestSparseBoost:
    """Boost sparse coverage."""

    def test_sparse_interpreter(self):
        interp = run_matlab("S = speye(5);")
        S = interp.global_env.get("S")
        assert S is not None

    def test_spzeros_interpreter(self):
        interp = run_matlab("S = spzeros(5, 5);")
        S = interp.global_env.get("S")
        assert S is not None

    def test_full_interpreter(self):
        interp = run_matlab("S = speye(5);\nF = full(S);")
        F = interp.global_env.get("F")
        assert isinstance(F, Mat)

    def test_nnz_interpreter(self):
        interp = run_matlab("S = speye(5);\nn = nnz(S);")
        n = interp.global_env.get("n")
        assert get_val(n) == 5

    def test_issparse_interpreter(self):
        interp = run_matlab("S = speye(5);\nr = issparse(S);")
        r = interp.global_env.get("r")
        assert r

    def test_sprand_interpreter(self):
        interp = run_matlab("S = sprand(5, 5, 0.5);")
        S = interp.global_env.get("S")
        assert S is not None

    def test_sprandn_interpreter(self):
        interp = run_matlab("S = sprandn(5, 5, 0.5);")
        S = interp.global_env.get("S")
        assert S is not None


class TestStatisticsBoost:
    """Boost statistics coverage."""

    def test_mean_interpreter(self):
        interp = run_matlab("x = [1 2 3 4 5];\nm = mean(x);")
        m = interp.global_env.get("m")
        assert abs(get_val(m) - 3.0) < 0.01

    def test_median_interpreter(self):
        interp = run_matlab("x = [1 2 3 4 5];\nm = median(x);")
        m = interp.global_env.get("m")
        assert abs(get_val(m) - 3.0) < 0.01

    def test_std_interpreter(self):
        interp = run_matlab("x = [1 2 3 4 5];\ns = std(x);")
        s = interp.global_env.get("s")
        assert get_val(s) > 0

    def test_var_interpreter(self):
        interp = run_matlab("x = [1 2 3 4 5];\nv = var(x);")
        v = interp.global_env.get("v")
        assert get_val(v) > 0

    def test_cov_interpreter(self):
        interp = run_matlab("x = [1 2 3 4 5];\ny = [2 4 6 8 10];\nc = cov(x, y);")
        c = interp.global_env.get("c")
        assert c is not None

    def test_corrcoef_interpreter(self):
        interp = run_matlab("x = [1 2 3 4 5];\ny = [2 4 6 8 10];\nr = corrcoef(x, y);")
        r = interp.global_env.get("r")
        assert r is not None

    def test_min_interpreter(self):
        interp = run_matlab("x = [3 1 4 1 5];\nm = min(x);")
        m = interp.global_env.get("m")
        assert get_val(m) == 1

    def test_max_interpreter(self):
        interp = run_matlab("x = [3 1 4 1 5];\nm = max(x);")
        m = interp.global_env.get("m")
        assert get_val(m) == 5

    def test_sum_interpreter(self):
        interp = run_matlab("x = [1 2 3 4 5];\ns = sum(x);")
        s = interp.global_env.get("s")
        assert get_val(s) == 15

    def test_prod_interpreter(self):
        interp = run_matlab("x = [1 2 3 4 5];\np = prod(x);")
        p = interp.global_env.get("p")
        assert get_val(p) == 120

    def test_cumsum_interpreter(self):
        interp = run_matlab("x = [1 2 3 4 5];\ny = cumsum(x);")
        y = interp.global_env.get("y")
        assert isinstance(y, Mat)

    def test_cumprod_interpreter(self):
        interp = run_matlab("x = [1 2 3 4 5];\ny = cumprod(x);")
        y = interp.global_env.get("y")
        assert isinstance(y, Mat)

    def test_diff_interpreter(self):
        interp = run_matlab("x = [1 3 6 10];\ny = diff(x);")
        y = interp.global_env.get("y")
        assert isinstance(y, Mat)

    def test_sort_interpreter(self):
        interp = run_matlab("x = [3 1 4 1 5];\ny = sort(x);")
        y = interp.global_env.get("y")
        assert isinstance(y, Mat)

    def test_unique_interpreter(self):
        interp = run_matlab("x = [1 2 2 3 3 3];\ny = unique(x);")
        y = interp.global_env.get("y")
        assert isinstance(y, Mat)

    def test_find_interpreter(self):
        interp = run_matlab("x = [0 1 0 1 0];\ni = find(x);")
        i = interp.global_env.get("i")
        assert isinstance(i, Mat)

    def test_length_interpreter(self):
        interp = run_matlab("x = [1 2 3 4 5];\nn = length(x);")
        n = interp.global_env.get("n")
        assert get_val(n) == 5

    def test_size_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\ns = size(A);")
        s = interp.global_env.get("s")
        assert s is not None

    def test_numel_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\nn = numel(A);")
        n = interp.global_env.get("n")
        assert get_val(n) == 4

    def test_isequal_interpreter(self):
        interp = run_matlab("x = [1 2 3];\ny = [1 2 3];\nr = isequal(x, y);")
        r = interp.global_env.get("r")
        assert r

    def test_isempty_interpreter(self):
        interp = run_matlab("x = [];\nr = isempty(x);")
        r = interp.global_env.get("r")
        assert r

    def test_isscalar_interpreter(self):
        interp = run_matlab("x = 42;\nr = isscalar(x);")
        r = interp.global_env.get("r")
        assert r

    def test_isvector_interpreter(self):
        interp = run_matlab("x = [1 2 3];\nr = isvector(x);")
        r = interp.global_env.get("r")
        assert r

    def test_ismatrix_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\nr = ismatrix(A);")
        r = interp.global_env.get("r")
        assert r

    def test_norm_interpreter(self):
        interp = run_matlab("v = [3 4];\nn = norm(v);")
        n = interp.global_env.get("n")
        assert abs(get_val(n) - 5.0) < 0.01

    def test_dot_interpreter(self):
        interp = run_matlab("a = [1 2 3];\nb = [4 5 6];\nd = dot(a, b);")
        d = interp.global_env.get("d")
        assert d is not None

    def test_cross_interpreter(self):
        interp = run_matlab("a = [1 0 0];\nb = [0 1 0];\nc = cross(a, b);")
        c = interp.global_env.get("c")
        assert isinstance(c, Mat)

    def test_trace_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\nt = trace(A);")
        t = interp.global_env.get("t")
        assert get_val(t) == 5

    def test_rank_interpreter(self):
        interp = run_matlab("A = [1 0; 0 1];\nr = rank(A);")
        r = interp.global_env.get("r")
        assert get_val(r) == 2

    def test_det_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\nd = det(A);")
        d = interp.global_env.get("d")
        assert abs(get_val(d) - (-2.0)) < 0.01

    def test_inv_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = inv(A);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_eig_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\n[V, D] = eig(A);")
        V = interp.global_env.get("V")
        D = interp.global_env.get("D")
        assert isinstance(V, Mat)
        assert isinstance(D, Mat)

    def test_svd_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\n[U, S, V] = svd(A);")
        U = interp.global_env.get("U")
        S = interp.global_env.get("S")
        V = interp.global_env.get("V")
        assert isinstance(U, Mat)
        assert isinstance(S, Mat)
        assert isinstance(V, Mat)

    def test_lu_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\n[L, U] = lu(A);")
        L = interp.global_env.get("L")
        U = interp.global_env.get("U")
        assert isinstance(L, Mat)
        assert isinstance(U, Mat)

    def test_qr_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\n[Q, R] = qr(A);")
        Q = interp.global_env.get("Q")
        R = interp.global_env.get("R")
        assert isinstance(Q, Mat)
        assert isinstance(R, Mat)

    def test_chol_interpreter(self):
        interp = run_matlab("A = [4 2; 2 3];\nL = chol(A);")
        L = interp.global_env.get("L")
        assert isinstance(L, Mat)

    def test_pinv_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = pinv(A);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_cond_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\nc = cond(A);")
        c = interp.global_env.get("c")
        assert c is not None

    def test_rref_interpreter(self):
        interp = run_matlab("A = [1 2 3; 4 5 6];\nR = rref(A);")
        R = interp.global_env.get("R")
        assert isinstance(R, Mat)

    def test_diag_interpreter(self):
        interp = run_matlab("D = diag([1 2 3]);")
        D = interp.global_env.get("D")
        assert isinstance(D, Mat)
        assert D.data.shape == (3, 3)

    def test_triu_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = triu(A);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_tril_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = tril(A);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_flipud_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = flipud(A);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_fliplr_interpreter(self):
        interp = run_matlab("A = [1 2; 3 4];\nB = fliplr(A);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_repmat_interpreter(self):
        interp = run_matlab("A = [1 2];\nB = repmat(A, 2, 3);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)

    def test_reshape_interpreter(self):
        interp = run_matlab("A = [1 2 3 4 5 6];\nB = reshape(A, 2, 3);")
        B = interp.global_env.get("B")
        assert isinstance(B, Mat)
        assert B.data.shape == (2, 3)

    def test_rand_interpreter(self):
        interp = run_matlab("R = rand(3, 4);")
        R = interp.global_env.get("R")
        assert isinstance(R, Mat)
        assert R.data.shape == (3, 4)

    def test_randn_interpreter(self):
        interp = run_matlab("R = randn(3, 4);")
        R = interp.global_env.get("R")
        assert isinstance(R, Mat)
        assert R.data.shape == (3, 4)

    def test_zeros_interpreter(self):
        interp = run_matlab("A = zeros(3, 4);")
        A = interp.global_env.get("A")
        assert isinstance(A, Mat)
        assert A.data.shape == (3, 4)

    def test_ones_interpreter(self):
        interp = run_matlab("A = ones(3, 4);")
        A = interp.global_env.get("A")
        assert isinstance(A, Mat)
        assert A.data.shape == (3, 4)

    def test_eye_interpreter(self):
        interp = run_matlab("A = eye(3);")
        A = interp.global_env.get("A")
        assert isinstance(A, Mat)
        assert A.data.shape == (3, 3)

    def test_linspace_interpreter(self):
        interp = run_matlab("x = linspace(0, 1, 5);")
        x = interp.global_env.get("x")
        assert isinstance(x, Mat)
        assert len(x.data) == 5

    def test_abs_interpreter(self):
        interp = run_matlab("x = abs(-5);")
        x = interp.global_env.get("x")
        assert get_val(x) == 5

    def test_sqrt_interpreter(self):
        interp = run_matlab("x = sqrt(16);")
        x = interp.global_env.get("x")
        assert get_val(x) == 4.0

    def test_exp_interpreter(self):
        interp = run_matlab("x = exp(0);")
        x = interp.global_env.get("x")
        assert abs(get_val(x) - 1.0) < 0.01

    def test_log_interpreter(self):
        interp = run_matlab("x = log(1);")
        x = interp.global_env.get("x")
        assert abs(get_val(x)) < 0.01

    def test_sin_interpreter(self):
        interp = run_matlab("x = sin(0);")
        x = interp.global_env.get("x")
        assert abs(get_val(x)) < 0.01

    def test_cos_interpreter(self):
        interp = run_matlab("x = cos(0);")
        x = interp.global_env.get("x")
        assert abs(get_val(x) - 1.0) < 0.01

    def test_tan_interpreter(self):
        interp = run_matlab("x = tan(0);")
        x = interp.global_env.get("x")
        assert abs(get_val(x)) < 0.01

    def test_asin_interpreter(self):
        interp = run_matlab("x = asin(0);")
        x = interp.global_env.get("x")
        assert abs(get_val(x)) < 0.01

    def test_acos_interpreter(self):
        interp = run_matlab("x = acos(1);")
        x = interp.global_env.get("x")
        assert abs(get_val(x)) < 0.01

    def test_atan_interpreter(self):
        interp = run_matlab("x = atan(0);")
        x = interp.global_env.get("x")
        assert abs(get_val(x)) < 0.01

    def test_ceil_interpreter(self):
        interp = run_matlab("x = ceil(3.2);")
        x = interp.global_env.get("x")
        assert get_val(x) == 4.0

    def test_floor_interpreter(self):
        interp = run_matlab("x = floor(3.8);")
        x = interp.global_env.get("x")
        assert get_val(x) == 3.0

    def test_round_interpreter(self):
        interp = run_matlab("x = round(3.7);")
        x = interp.global_env.get("x")
        assert get_val(x) == 4.0

    def test_fix_interpreter(self):
        interp = run_matlab("x = fix(3.8);")
        x = interp.global_env.get("x")
        assert get_val(x) == 3.0

    def test_sign_interpreter(self):
        interp = run_matlab("x = sign(-5);")
        x = interp.global_env.get("x")
        assert get_val(x) == -1

    def test_mod_interpreter(self):
        interp = run_matlab("x = mod(7, 3);")
        x = interp.global_env.get("x")
        assert get_val(x) == 1

    def test_rem_interpreter(self):
        interp = run_matlab("x = rem(7, 3);")
        x = interp.global_env.get("x")
        assert get_val(x) == 1
