function [f,g] = float_neg_loglikelihood(y, H, x, D, sigma)    
    [m,n] = size(H);
    q =@(z) sigma^2*z + H*x + (D.*coth(D.*(z*x')))*x - n*ones(m,1)./z - y;
    qp=@(z) sigma^2*ones(m,1) - ((D.*csch(D.*(z*x'))).^2)*(x.^2) + n./(z.^2);
    
    a = -1e6*ones(m,1)+1e-3;
    b = 1e6*ones(m,1);
    
    t = newton_safe(q, qp, a, b);
    
    A = D.*(t*x');
    V = log(sinh(A) ./ A);
    idx = (abs(V)==Inf);
    V(idx) = abs(A(idx)) - log(abs(2*A(idx))); % is this the correct Taylor expansion??? Double check 
    K = 0.5*sigma^2*(t.^2) + t.*(H*x) + V*ones(n,1);
    Kpp = qp(t);
  
    f = -(ones(1,m)*(K - 0.5*log(Kpp)) - t'*y);
    if nargout > 1
        g = -log_likelihood_grad(x,t,y,H,sigma,D);
    end
end




function grad_l = log_likelihood_grad(x,t,y,H,sigma,D)
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
    t_sq = t.^2;
    t_cb = t.*t_sq;
    Hx = H*x;
    A = D.*(t*x');
    
    DcothA = D.*coth(A);
    DcschA = D.*csch(A);
    DcschA_sq = DcschA.^2;
    
    Kp = sigma^2*t + Hx + DcothA*x - n./t;
    Kpp = sigma^2*ones(m,1) - DcschA_sq*(x.^2) + n ./ t_sq;
    Kppp = 2*(DcothA.*DcschA_sq)*x_cb - 2*n./t_cb;
    
    dK_dx = t*ones(1,n) .* (H + DcothA) - ones(m,1)./x';
    dKp_dx = H + DcothA - (t*x').*DcschA_sq;
    dKpp_dx = 2*DcschA_sq .* ((t*x_sq') .* DcothA - ones(m,1)*x'); 
    
    dl_dx    = ones(1,m)*(dK_dx - 0.5*dKpp_dx./Kpp);
    dl_dt    = Kp' - 0.5*(Kppp ./ Kpp)' - y';
    dq_dt_inv= 1./Kpp; % here we just store the diagonal elements, not mat
    dq_dx    = dKp_dx;
    
    grad_l = (dl_dx - (dl_dt.*dq_dt_inv')*dq_dx)';
end





