clear; clc;
rng(1);

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

S = load(fullfile(this_dir, 'data_sample.mat'));
X = double(S.X);
Y_label = double(S.Y_label(:));

k = numel(unique(Y_label));
n = size(X, 2);
Y = zeros(k, n);
for i = 1:n
    Y(Y_label(i), i) = 1;
end

lambda = 0.01;
alpha = 0.01;
beta = 0.01;
max_iter = 15;
top_k = 24;
knn_graph = 15;

[W, Theta, b, accuracy, redundancy, obj_history] = MRRM(X, Y, Y_label, ...
    lambda, alpha, beta, max_iter, top_k, knn_graph);

[~, rank] = sort(diag(Theta), 'descend');
sel = rank(1:min(top_k, size(X, 1)));

fprintf('samples      : %d\n', n);
fprintf('features     : %d\n', size(X, 1));
fprintf('classes      : %d\n', k);
fprintf('accuracy     : %.4f\n', accuracy);
fprintf('redundancy   : %.4f\n', redundancy);
fprintf('iterations   : %d\n', numel(obj_history));
fprintf('objective    : %.6g\n', obj_history(end));
fprintf('top-%d features: %s\n', top_k, mat2str(sel(:)'));
