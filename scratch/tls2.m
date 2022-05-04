function x = tls2(A, b, W)
    n = size(A,2);
    W = diag(1./diag(W));
    %construct weighted matrix 
    C = [A,b]*W;
    
    % find right singular vectors of 
    [~,~,V] = svd(C,'econ');
    
    % take smallest right singular vector
    q = V(:,n+1);
    
    % transform 
    v = W*q;
    x = -(1/v(n+1))*v(1:n);
    

end
