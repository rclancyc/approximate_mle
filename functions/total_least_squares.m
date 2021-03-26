function x_tls  = total_least_squares(A, b)
    % [x_tls, A_tls] = totalLeastSquaresRegression(A, b)
    %INPUTS   
    %   A: Data matrix
    %   b: signal or measurement vector (i.e. right hand side)
    %RETURNS
    %   x_tls: Solution to total least squares problem give below
    %EXAMPLE: 
    %    x_tls = totalLeastSquaresRegression(randn(10,5),rand(10,1));
    %
    % solves the problem min {||[E,r]||_F} subject to (A+E)x = b+r.
    % A and E are (m x n) matrices, b and r are (m x 1) vector. [E,r] is 
    % an (m x (n+1)) uncertainty matrix where E and r represent uncertainty 
    % in the data matrix and RHS respectively. 
    % 
    % Richie Clancy 05/04/2020
    
    [~, n] = size(A);
    C = [A, b];
    [~,~,V] = svd(C,'econ');
    v = V(:,n+1);
    x_tls = -(1/v(n+1))*v(1:n);

end