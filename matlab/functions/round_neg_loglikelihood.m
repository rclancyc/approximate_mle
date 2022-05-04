function [f,g] = round_neg_loglikelihood(y, H, x, c, sigma)
    
    

    [m,n] = size(H);
    q =@(z) sigma^2*z + H*x + c*coth(c*z*x')*x - n*ones(m,1)./z - y;
    qp=@(z) sigma^2*ones(m,1) - c^2*(csch(c*z*x').^2)*(x.^2) + n./(z.^2);
    
    % Establish upper and lower bounds for bracketed Newton's method.
    % Small bump in a gets rid of numerical issues with initializing t = 0
    a = -1e5*ones(m,1)+1e-3;
    b = 1e5*ones(m,1);
    
    t = newton_safe(q, qp, a, b);
    
    M = c*t*x';
    V = log(sinh(M)./M);
    
    % address overflow issues with Taylor approximation
    idx = (abs(V)==Inf);
    V(idx) = abs(M(idx)) - log(abs(2*M(idx)));  % is this the correct taylor expansion??? Double check
    % Note that this is supposed to be a Taylor approximation but seems to be incorrect.
    % When t is small for uniform noise, we have the following
    % K_unif(dtx) = 
    
    
    % set cumulant generating funciton vector and its 2nd derivative
    K = 0.5*sigma^2*(t.^2) + t.*(H*x) + V*ones(n,1);
    Kpp = qp(t);
  
    f = -(ones(1,m)*(K - 0.5*log(Kpp)) - t'*y);
    if nargout > 1
        g = -log_likelihood_grad(x,t,y,H,sigma,c);
    end
end




function grad_l = log_likelihood_grad(x,t,y,H,sigma,c)
    % use chain rule and implicit differentiation to calculate the 
    % derivative. In particular, we want to differentiate df(x,t(x))/dx 
    % where t is related to x via the equation q(x,t) = 0, then we have:
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
    t_sq = t.^2;
    t_cb = t.*t_sq;
    Hx = H*x;
    ctx = c*t*x';
    coth_ctx = coth(ctx);
    csch_ctx_sq = csch(ctx).^2;
    
    Kp      = sigma^2*t + Hx + c*coth_ctx*x - n./t ;
    Kpp     = sigma^2*ones(m,1) - c^2*csch_ctx_sq*x_sq + n./t_sq; 
    Kppp    = 2*c^3*(coth_ctx.*csch_ctx_sq)*x_cb - 2*n./t_cb;
    
    dK_dx    = t*ones(1,n).*(H + c*coth_ctx) - ones(m,1)./x'; 
    dKp_dx   = H + c*coth_ctx - (c*ctx.*csch_ctx_sq);
    dKpp_dx  = 2*c^2*csch_ctx_sq.*((c*t*x_sq').*coth_ctx - ones(m,1)*x');
    
    dl_dx    = ones(1,m)*(dK_dx - 0.5*dKpp_dx./Kpp);
    dl_dt    = Kp' - 0.5*(Kppp ./ Kpp)' - y';
    dq_dt_inv= 1./Kpp; 
    dq_dx    = dKp_dx;
    
    grad_l = (dl_dx - (dl_dt.*dq_dt_inv')*dq_dx)';
end
