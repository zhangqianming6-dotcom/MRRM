function [W, Theta, b, accuracy, redundancy, obj_history] = MRRM(X, Y, Y_label, lambda, alpha, beta, max_iter, top_k, knn_graph)
if nargin < 9 || isempty(knn_graph)
    knn_graph = 15;
end

[d, n] = size(X);
k = size(Y, 1);

theta_vec = ones(d, 1) / d;
Theta = diag(theta_vec);
W = orth(randn(d, k));
b = zeros(k, 1);

A = redundancy_matrix(X);
L = graph_laplacian(X, knn_graph);
H = eye(n) - ones(n) / n;

XHXt = X * H * X';
XLXt = X * L * X';
XHYt = X * H * Y';

W_prev = W;
obj_history = zeros(max_iter, 1);

for iter = 1:max_iter
    b = mean(Y, 2) - W' * Theta * mean(X, 2);

    A_w = Theta * XHXt * Theta + alpha * (Theta * XLXt * Theta) + beta * eye(d);
    B_w = Theta * XHYt;
    W = gpi_solve(A_w, B_w, W);

    P = W * W';
    Q_total = P .* XHXt + alpha * (P .* XLXt) + lambda * A;
    s_vec = 2 * sum(W .* XHYt, 2);
    theta_vec = alm_solve_theta(Q_total, s_vec, theta_vec);
    Theta = diag(theta_vec);

    R = W' * Theta * X + b * ones(1, n) - Y;
    obj_history(iter) = norm(R, 'fro')^2 ...
        + beta * norm(W, 'fro')^2 ...
        + alpha * trace(W' * Theta * XLXt * Theta * W) ...
        + lambda * (theta_vec' * A * theta_vec);

    if iter > 1 && norm(W - W_prev, 'fro') < 1e-6
        break;
    end
    W_prev = W;
end

obj_history = obj_history(1:iter);

b = mean(Y, 2) - W' * Theta * mean(X, 2);
scores = W' * Theta * X + b * ones(1, n);
[~, Y_pred] = max(scores, [], 1);
Y_pred = Y_pred';
accuracy = sum(Y_label(:) == Y_pred(:)) / numel(Y_label);

[~, selected] = sort(diag(Theta), 'descend');
top_features = selected(1:min(top_k, d));
m = numel(top_features);
if m < 2
    redundancy = 0;
else
    sub_A = A(top_features, top_features);
    redundancy = (sum(sub_A(:)) - trace(sub_A)) / (m * (m - 1));
end
end

function A = redundancy_matrix(X)
n = size(X, 2);
H = eye(n) - ones(n) / n;
F = X * H;
norms = max(sqrt(sum(F.^2, 2)), eps);
F = F ./ norms;
A = (F * F') .^ 2;
end

function L = graph_laplacian(X, knn)
n = size(X, 2);
Dist2 = zeros(n);
for i = 1:n
    Dist2(i, :) = sum((X - X(:, i)).^2, 1);
end

knn_dists = zeros(n, knn);
for i = 1:n
    [~, idx] = sort(Dist2(i, :));
    knn_dists(i, :) = Dist2(i, idx(2:knn + 1));
end

sigma2 = median(knn_dists(:));
if sigma2 < eps
    sigma2 = eps;
end

S = zeros(n);
for i = 1:n
    d_i = Dist2(i, :);
    [~, idx] = sort(d_i);
    nb = idx(2:knn + 1);
    w = exp(-d_i(nb) / (2 * sigma2));
    S(i, nb) = w;
    S(nb, i) = w;
end

deg = sum(S, 2);
d_inv_sqrt = 1 ./ sqrt(max(deg, eps));
L = eye(n) - (d_inv_sqrt .* S) .* d_inv_sqrt';
end

function W = gpi_solve(A_w, B_w, W)
max_iter = 100;
tol = 1e-8;
shift = max(real(eig(A_w))) + 1;
A_hat = shift * eye(size(A_w)) - A_w;

for t = 1:max_iter
    Z = 2 * A_hat * W + 2 * B_w;
    [U, ~, V] = svd(Z, 'econ');
    W_new = U * V';
    if norm(W_new - W, 'fro') < tol
        W = W_new;
        break;
    end
    W = W_new;
end
end

function theta = alm_solve_theta(Q, s, theta_init)
max_iter = 50;
tol = 1e-8;
rho = 1.2;
mu = 1.0;

d = numel(theta_init);
theta = theta_init(:);
v = theta;
eta = zeros(d, 1);
gamma = 0;
ones_d = ones(d, 1);

for iter = 1:max_iter
    lhs = 2 * Q + mu * eye(d) + mu * (ones_d * ones_d');
    rhs = s - eta + mu * v - gamma * ones_d + mu * ones_d;
    theta_new = lhs \ rhs;

    v = max(theta_new + eta / mu, 0);
    eta = eta + mu * (theta_new - v);
    gamma = gamma + mu * (sum(theta_new) - 1);
    mu = rho * mu;

    if norm(theta_new - theta) < tol
        theta = theta_new;
        break;
    end
    theta = theta_new;
end

theta = max(theta, 0);
if sum(theta) > 0
    theta = theta / sum(theta);
else
    theta = ones(d, 1) / d;
end
end
