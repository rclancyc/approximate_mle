clear;
addpath(genpath('functions'))

opts.Display = 'off';
opts.Verbose = 'on';


n_sims = 10000;
M = 55;
N = 50;
model_names = {'Exponential clipping', 'Gaussian', 'Rounding', 'Floating point'};

% first dim for simulation, second for method, third for model, i.e.m
% sim = 100, 1 for MLE, 3 for rounding)
err_array = zeros(n_sims, 3, 4);

for model = 1:4
    if opts.Verbose
        fprintf('\n \n \n %s \n', model_names{model})
        fprintf('\t || \t AMLE Error \t || \t OLS Error \t || \t TLS Error \n')
        fprintf('============================================================================ \n')
    end   
    for i = 1:n_sims
        % set seed for each simulation
        rng(i);

        % generate true solution from which data is generated
        u = rand(N, 1);                 
        xtrue = tan(pi*(u-0.5));

        sigma = 0.1;
        eta = sigma*randn(M,1);
        switch model
            case 1 % exponential clipping
                lambda = 2;         % exponential of rate lambda has mean 1/lambda
                gamma = 2;          % smaller means clipped occurs sooner                
                U = rand(M,N);
                G = -(1/lambda)*log(1 - U).*sign(randn(M,N));
                H = sign(G).*min(abs(G),gamma);
                ytrue = G*xtrue;
                y = ytrue + eta;
                neg_L =@(z) clipped_exp_neg_loglikelihood(y, H, z, sigma, lambda, gamma);
            case 2 % gaussian
                rho = 2;            % standard deviation of Gaussian noise for design matrix
                H = 10*(randn(M,N));
                G = H + rho*randn(M,N);
                ytrue = G*xtrue;
                y = ytrue + eta;                
                neg_L =@(z) gaussian_neg_loglikelihood(y, H, z, rho, sigma);
            case 3 % rounding
                rtd = 0;            % round to digit, 0 is ones spot, 1 is tenths, etc
                c = 0.5*10^(-rtd);  % uncertainty parameter, i.e. g ~ unif(h - c, h + c)
                G = 10*(rand(M,N)); 
                H = round(G,rtd);
                ytrue = G*xtrue;
                y = ytrue + eta;                
                neg_L =@(z) round_neg_loglikelihood(y, H, z, c, sigma);
            case 4 % floating point
                sig_figs = 2;       % number of significant figures to use
                dyn_range = 3;      % large dyn_range increases variability of exponents for design                
                G = randn(M,N).*10.^(randi([0,dyn_range], M,N));
                D = floor(log10(abs(G)));
                H = round(G./ 10.^D, sig_figs-1).* 10.^D;
                D = 5*10.^(D-sig_figs);
                ytrue = G*xtrue;
                y = ytrue + eta;                
                neg_L =@(z) float_neg_loglikelihood(y, H, z, D, sigma);
        end

        xols = H \ y;
        xtls = total_least_squares(H, y);
        xmle = minFunc(neg_L,xols,opts);

        err_array(i,1,model) = norm(xmle - xtrue)/norm(xtrue);
        err_array(i,2,model) = norm(xols - xtrue)/norm(xtrue);
        err_array(i,3,model) = norm(xtls - xtrue)/norm(xtrue);
        if opts.Verbose
            fprintf('Sim %i \t || \t %4.6f \t || \t %4.6f \t || \t %4.6f \n', ...
                i, err_array(i,1,model), err_array(i,2,model), err_array(i,3,model))      
        end
    end
        
end

save('data/data_for_histograms', 'err_array', 'M', 'N', 'model_names')



%%
plot_fixed_row_and_column_count




