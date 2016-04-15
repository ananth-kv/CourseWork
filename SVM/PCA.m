function [E, V] = PCA(X,K)

mu = mean(X);
N = size(X,1);
X = X - ones(N,1)*mu;
[~, ~, V] = svd(X);
V = V(:,1:K);
E = X*V;

end