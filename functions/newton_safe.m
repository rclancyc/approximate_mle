function myroot = newton_safe(f, df, a, b)
    % myroot = newton_safe(f, df, a, b)
    %
    % INPUTS
    % f: function handle on which to find root
    % df: function gradient handle
    % a: left bracket for bounding
    % b: right bracket for bounding
    %
    % OUTPUTS
    % myroot: a root for function f
    %
    % This function performs a bracketed Newton Method that falls back to
    % the Bisection method if a step is taken outside of the current
    % interval for which a root exists. This particular implementation acts
    % on a vector valued function that has a diagonal Jacobian and is
    % purpose built finding t(x) in the approximate MLE problem 
 
    
    tol = 1e-8;
    maxit = 200;
    
    % set initial t based on bounds and small perturbation
    t = 0.5*(a+b);
    
    it = 0;
    ft = f(t);
    dft = df(t); 
    while norm(ft, 'Inf') > tol && it < maxit
        it = it + 1;                                       
        [a,b] = bisect_interval(f,a,b);     % bracket interval each time to ensure progress
        t = t - ft./dft;                    % take newton step               
        idx = ((t <= a) | (t >= b));        % find index of iterates outside of safety brackets
        t(idx) = (a(idx) + b(idx))/2;       % choose midpoint for failed newton steps
        ft = f(t);                          % set new function value
        dft = df(t);                        % set new gradient value
    end
    
    myroot = t;
    
end



function [a, b] = bisect_interval(f, a, b)
    % [a, b] = bisect_interval(f, a, b)
    %
    % INPUTS
    % f: function handle
    % a: lower boundary for a hypercube containing a root of f
    % b: upper boundary for a hypercube containing a root of f\
    %
    % OUTPUTS
    % a: new lower boundary after interval has been bisected
    % b: new upper boundary after interval has been bisected
    
    c  = 0.5*(a+b);
    idx = (f(a).*f(c) > 0);
    a(idx) = c(idx);
    b(~idx) = c(~idx);
end




