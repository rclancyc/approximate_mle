function x = tls3(A, b)
    n = size(A,2);
    
    [~,~,V] = svd([A,b],'econ');
    x = -(1/V(n+1,n+1))*V(1:n,n+1);
end
