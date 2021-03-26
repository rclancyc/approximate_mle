clear;

opts.Display = 'off';
opts.Verbose = 'on';

n_sims = 1000;
M = 100;            % Max number of rows 

ns = round(logspace(0,2,8));
ns(end) = M-1;
N = max(ns);            % Max number of rows

median_error_array = zeros(length(ns), 3, 4);
mean_error_array = zeros(length(ns), 3, 4);
median_error_ratio = zeros(length(ns), 3, 4);
mean_error_ratio = zeros(length(ns), 3, 4);

model_names = {'Exponential clipping', 'Gaussian', 'Rounding', 'Floating point'};

for model = 1:4
    % initialize arrays to store experimental details
    solution_array = cell(n_sims,4);
    G_array = cell(n_sims,1);
    H_array = cell(n_sims,1);
    D_array = cell(n_sims,1);
    true_solution_array = cell(n_sims,1);
    eta_array = cell(n_sims,1);

    xols_error = zeros(n_sims, length(ns));
    xtls_error = zeros(n_sims, length(ns));
    xmle_error = zeros(n_sims, length(ns));

    xols_res = zeros(n_sims, length(ns));
    xtls_res = zeros(n_sims, length(ns));
    xmle_res = zeros(n_sims, length(ns));


    fprintf('For %i %s simulations for %i column matrices \n', n_sims, model_names{model}, N); 
    for i = 1:n_sims
        % set seed for each simulation
        rng(i);

        % generate true solution from which data is generated
        u = rand(N, 1);
        xtrue = tan(pi*(u-0.5));

        sigma = 0.1;
        switch model
            case 1 % exponential clipping
                lambda = 2;         % exponential of rate lambda has mean 1/lambda
                gamma = 2;          % smaller means clipped occurs sooner                
                U = rand(M,N);
                G = -(1/lambda)*log(1 - U).*sign(randn(M,N));
                H = sign(G).*min(abs(G),gamma);
            case 2 % gaussian
                rho = 2;            % standard deviation of Gaussian noise for design matrix
                H = 10*(randn(M,N));
                G = H + rho*randn(M,N);
            case 3 % rounding
                rtd = 0;            % round to digit, 0 is ones spot, 1 is tenths, etc
                c = 0.5*10^(-rtd);  % uncertainty parameter, i.e. g ~ unif(h - c, h + c)
                G = 10*(rand(M,N)); 
                H = round(G,rtd);
            case 4 % floating point
                sig_figs = 2;       % number of significant figures to use
                dyn_range = 3;      % large dyn_range increases variability of exponents for design                
                G = randn(M,N).*10.^(randi([0,dyn_range], M,N));
                D = floor(log10(abs(G)));
                H = round(G./ 10.^D, sig_figs-1).* 10.^D;
                D = 5*10.^(D-sig_figs);
                D_array{i} = D;
        end


        eta = sigma*randn(M,1);

        % store data to be used again
        eta_array{i} = eta;
        G_array{i} = G;
        H_array{i} = H;
        true_solution_array{i} = xtrue;
    end


    for jj = 1:length(ns)
        n = ns(jj);
        fprintf(' \n \n \n \n \n \nNUMBER OF COLUMNS  = %i', n)
        % main loop, each ii is new instance of experiment
        
        if opts.Verbose
            fprintf('\n \n \n %s:  number of columns %i \n', model_names{model}, ns(jj))
            fprintf('\t || \t AMLE Error \t || \t OLS Error \t || \t TLS Error \n')
            fprintf('============================================================================ \n')
        end        
        
        tic
        for ii = 1:n_sims
            
            % retrieve desired simulation run and take desired number of columns
            xtrue = true_solution_array{ii}(1:n); 
            xtrue = xtrue/norm(xtrue);
            
            H = H_array{ii}(:,1:n);
            G = G_array{ii}(:,1:n);    
            eta = eta_array{ii};

            % create right hand side 
            y = G*xtrue + eta;

            % solve least squares and total least squares problems
            xols = H \ y;
            xtls = total_least_squares(H, y);

            % set objective function depending on model
            switch model
                case 1
                    neg_L =@(z) clipped_exp_neg_loglikelihood(y, H, z, sigma, lambda, gamma);
                case 2
                    neg_L =@(z) gaussian_neg_loglikelihood(y, H, z, rho, sigma);
                case 3
                    neg_L =@(z) round_neg_loglikelihood(y, H, z, c, sigma);
                case 4
                    D = D_array{ii}(:,1:n);
                    neg_L =@(z) float_neg_loglikelihood(y, H, z, D, sigma);
            end     


            x0 = xols;

            xmle = minFunc(neg_L,x0,opts);

            % store errors
            xols_error(ii, jj) = norm(xols  - xtrue);
            xtls_error(ii, jj) = norm(xtls - xtrue);
            xmle_error(ii, jj) = norm(xmle - xtrue);

            %store residuals
            xols_res(ii, jj) = norm(G*xols  - y);
            xtls_res(ii, jj) = norm(G*xtls  - y);
            xmle_res(ii, jj) = norm(G*xmle  - y);
            if opts.Verbose
                fprintf('Sim %i \t || \t %4.6f \t || \t %4.6f \t || \t %4.6f \n', ...
                    ii, norm(xmle  - xtrue), norm(xols  - xtrue), norm(xtls - xtrue))      
            end
        end
        toc
    end
    
    % save all simulation data to avoid rerunning
    save_string = strcat('data/vary_cols_',model_names{model}, '.mat');
    save(save_string)
    
    % generate arrays for plots later 
    median_error_ratio(:,1,model) = median(xols_error ./ xmle_error);
    median_error_ratio(:,2,model) = median(xtls_error ./ xmle_error);
    
    mean_error_ratio(:,1,model) = mean(xols_error ./ xmle_error);
    mean_error_ratio(:,2,model) = mean(xtls_error ./ xmle_error);
    
    median_error_array(:, 1, model) = median(xmle_error);
    median_error_array(:, 2, model) = median(xols_error);
    median_error_array(:, 3, model) = median(xtls_error);
    
    mean_error_array(:, 1, model) = mean(xmle_error);
    mean_error_array(:, 2, model) = mean(xols_error);
    mean_error_array(:, 3, model) = mean(xtls_error);
end
    
% save data for plots
save('data/vary_cols_error_array.mat', 'mean_error_array', ...
    'median_error_array', 'median_error_ratio', 'mean_error_ratio','ns', 'model_names')


%%

plot_vary_by_cols_combined

