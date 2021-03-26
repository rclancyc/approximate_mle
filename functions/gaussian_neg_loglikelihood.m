function [f,g] = gaussian_neg_loglikelihood(y, H, x, rho, sigma)
    m = length(y);
    k = sigma^2 + rho^2*norm(x)^2;
    Hx_y = H*x-y;
    norm_Hx_y = norm(Hx_y);
    
    f = -1*(-0.5*(norm_Hx_y^2/k + m*log(k)));
    g = -1*((1/k)*(rho^2*(norm_Hx_y^2/k - m)*x - H'*Hx_y));
end