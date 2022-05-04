function [f,g] = clipped_exp_neg_loglikelihood(y, H, x, sigma, lambda, gamma)    
    [m,n] = size(H);    
    
    bLam = lambda*ones(m,n);
    A = (abs(H) == abs(gamma));
    S = sign(H);
    
    % Identify locations of singularities in approx. log-likelihood then
    % use inner-most about zero to set lower (a) and upper (b) bounds for 
    % bracketed Newton's method
    T = lambda*A.*S./(ones(m,1)*x');
    pidx = (T > 0);
    nidx = (T < 0);  
    temp = T;
    temp(~nidx) = -Inf;
    a = max(temp,[],2);
    a = a + 1e-4;
    a(a == -Inf) = -1e6;
    temp = T; 
    temp(~pidx) = Inf;
    b = min(temp,[],2);
    b = b - 1e-4;
    b(b == Inf) = 1e6;
    
    % define constaint and its derivative then use to find t
    q =@(z) sigma^2*z + H*x + (A.*S./(bLam - S.*(z*x')))*x - y;
    qp=@(z) sigma^2*ones(m,1) + (A ./ (bLam - S.*(z*x')).^2)*x.^2;
    t = newton_safe(q, qp, a,b);
    
    % define CGF and its 2nd derivative
    M = bLam ./ (bLam - S.*(t*x'));
    K = 0.5*sigma^2*(t.^2) + t.*(H*x) + (A .* log(M))*ones(n,1);
    Kpp = qp(t);
    
    % calculate approximate log-likelihood function and gradient
    f = -(ones(1,m)*(K - 0.5*log(Kpp)) - t'*y);
    if nargout > 1
        g = -log_likelihood_grad(y, H, A, x, t, sigma, lambda);
    end
    
end




function grad_l = log_likelihood_grad(y, H, A, x, t, sigma, lambda) 
    % use the adjoint state method or more realistically the chain rule and
    % implicit differentiation to calculate the derivative. In particular, 
    % we want to differentiate df(x,t(x))/dx where t is related to x via 
    %the equation q(x,t) = 0, then we have:
    %
    % grad_{wrt x} l =  dl/dx - (dl/dt)(dq/dt)^{-1}(dq/dx)     (eq. 1)
    %     (1xn)      =  (1xn)  -  (1xm)    (mxm)    (mxn)   
    %
    % This follows from the chain rule i.e. df/dx = df/dx + (df/dt)(dt/dx).
    % To find dt/dx, note that q=0 ==> dq/dx=0 ==> dq/dx+(dq/dt)(dt/dx)=0.
    % Solving for dt/dx gives dt/dx = -(dq/dt)^{-1}(dq/dx) hence (eq. 1).
    [m,n] = size(H);
    x_sq = x.^2;
    x_cb = x.*x_sq;
    S = sign(H);
    C = S.*(t*x');
    F = lambda*ones(m,n) - C;
    
    Kp      = sigma^2*t + H*x + (A.*S./F)*x;
    Kpp     = sigma^2*ones(m,1) + (A./F.^2)*x_sq;
    Kppp    = 2*(A.*S./F.^3)*x_cb; 
    
    
    dK_dx   = (t*ones(1,n)).*(H + A.*S./F);
    dKp_dx  = H + A.*(S./F + (t*x')./F.^2);
    dKpp_dx = 2*A.*((ones(m,1)*x')./F.^2 + S.*(t*x_sq')./F.^3);
    
    dl_dx       = ones(1,m)*(dK_dx - 0.5*dKpp_dx./Kpp);
    dl_dt       = Kp' - 0.5*(Kppp ./ Kpp)' - y';
    dq_dt_inv   = 1./Kpp; 
    dq_dx       = dKp_dx;
    
    grad_l = (dl_dx - (dl_dt.*dq_dt_inv')*dq_dx)';
end