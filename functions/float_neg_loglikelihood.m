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
    V(idx) = abs(A(idx)) - log(abs(2*A(idx))); 
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
    
    
%     Kp      = sigma^2*t + Hx + (D.*coth_A)*x - n./t ;
%     Kpp     = sigma^2*ones(m,1) - (D.^2.*csch_A_sq)*x_sq + n./t_sq; 
%     Kppp    = 2*(D.^3.*coth_A.*csch_A_sq)*x_cb - 2*n./t_cb;
%     
%     dK_dx    = t*ones(1,n).*(H + D.*coth_A) - ones(m,1)./x'; 
%     dKp_dx   = H + D.*coth_A - (D.*A.*csch_A_sq);
%     dKpp_dx  = 2*(D.^2.*csch_A_sq).*(D.*(t*x_sq').*coth_A - ones(m,1)*x');
    
    dl_dx    = ones(1,m)*(dK_dx - 0.5*dKpp_dx./Kpp);
    dl_dt    = Kp' - 0.5*(Kppp ./ Kpp)' - y';
    dq_dt_inv= 1./Kpp; % here we just store the diagonal elements, not mat
    dq_dx    = dKp_dx;
    
    grad_l = (dl_dx - (dl_dt.*dq_dt_inv')*dq_dx)';
end














% 
% function [f,g] = neg_loglikelihood_float(y, H, x, C, sigma)    
%     [m,n] = size(H);
%     q =@(z) sigma^2*z + H*x + (C.*coth(C.*(z*x')))*x - n*ones(m,1)./z - y;
%     qp=@(z) sigma^2*ones(m,1) - ((C.*csch(C.*(z*x'))).^2)*(x.^2) + n./(z.^2);
%     t = newton_safe(q, qp, 0.0005*ones(m,1));
%     
%     M = C.*(t*x');
%     V = log(sinh(M) ./ M);
%     idx = (abs(V)==Inf);
%     V(idx) = abs(M(idx)) - log(abs(2*M(idx))); 
%     K = 0.5*sigma^2*(t.^2) + t.*(H*x) + V*ones(n,1);
%     Kpp = qp(t);
%   
%     f = -(ones(1,m)*(K - 0.5*log(Kpp)) - t'*y);
%     if nargout > 1
%         g = -log_likelihood_grad(x,t,y,H,sigma,C);
%     end
% end
% 
% 
% 
% 
% function grad_l = log_likelihood_grad(x,t,y,H,sigma,C)
%     % use the adjoint state method or more realistically the chain rule and
%     % implicit differentiation to calculate the derivative. In particular, 
%     % we want to differentiate df(x,t(x))/dx where t is related to x via 
%     %the equation q(x,t) = 0, then we have:
%     %
%     % grad_{wrt x} l =  dl/dx - (dl/dt)(dq/dt)^{-1}(dq/dx)     (eq. 1)
%     %     (1xn)      =  (1xn)  -  (1xm)    (mxm)    (mxn)   
%     %
%     % This follows from the chain rule i.e. df/dx = df/dx + (df/dt)(dt/dx).
%     % To find dt/dx, note that q=0 ==> dq/dx=0 ==> dq/dx+(dq/dt)(dt/dx)=0.
%     % Solving for dt/dx gives dt/dx = -(dq/dt)^{-1}(dq/dx) hence (eq. 1).
%     [m,n] = size(H);
%     x_sq = x.^2;
%     x_cb = x.*x_sq;
%     t_sq = t.^2;
%     t_cb = t.*t_sq;
%     Hx = H*x;
%     ctx = D.*(t*x');
%     coth_ctx = coth(ctx);
%     csch_ctx_sq = csch(ctx).^2;
%     
%     Kp      = sigma^2*t + Hx + (C.*coth_ctx)*x - n./t ;
%     Kpp     = sigma^2*ones(m,1) - (C.^2.*csch_ctx_sq)*x_sq + n./t_sq; 
%     Kppp    = 2*(C.^3.*coth_ctx.*csch_ctx_sq)*x_cb - 2*n./t_cb;
%     
%     dK_dx    = t*ones(1,n).*(H + C.*coth_ctx) - ones(m,1)./x'; 
%     dKp_dx   = H + C.*coth_ctx - (C.*ctx.*csch_ctx_sq);
%     dKpp_dx  = 2*(C.^2.*csch_ctx_sq).*(C.*(t*x_sq').*coth_ctx - ones(m,1)*x');
%     
%     dl_dx    = ones(1,m)*(dK_dx - 0.5*dKpp_dx./Kpp);
%     dl_dt    = Kp' - 0.5*(Kppp ./ Kpp)' - y';
%     dq_dt_inv= 1./Kpp; % here we just store the diagonal elements, not mat
%     dq_dx    = dKp_dx;
%     
%     grad_l = (dl_dx - (dl_dt.*dq_dt_inv')*dq_dx)';
% end