
n=1000;
X = randn(n, 100);
xi = X(n, 1);
y = sum(X, 2)/n;

cvx_solver mosek
cvx_solver SDPT3
cvx_begin
variable x(n, 1)
maximize((x - xi)'*rand(n, 1));
subject to 
1 + y'*x >= (sqrt(2*n).*diag(n))*x;
1 + y'*x >= 1 - y'*x;
1 + y'*x >= 0;
cvx_end        
