clear;
addpath('functions') 


opts.Display = 'off';
opts.Verbose = 'on';

% set number of simulations, columns, and rows to use
n_sims = 5;        
N = 20;             
ms = round(logspace(log10(N),log10(N)+2,11));
M = max(ms);            % Max number of rows
ms(1) = N+1;

median_error_array = zeros(length(ms), 3, 4);
mean_error_array = zeros(length(ms), 3, 4);
model_names = {'Exponential clipping', 'Gaussian', 'Rounding', 'Floating point'};
for model = 1:4
    % 1: exponential clipping
    % 2: gaussian
    % 3: rounding
    % 4: floating_point

    % initialize arrays to store experimental details
    solution_array = cell(n_sims,4);
    rhs_array = cell(n_sims,1);
    G_array = cell(n_sims,1);
    H_array = cell(n_sims,1);
    D_array = cell(n_sims,1);
    true_solution_array = cell(n_sims,1);
    xols_error = zeros(n_sims, length(ms));
    xtls_error = zeros(n_sims, length(ms));
    xmle_error = zeros(n_sims, length(ms));
    xols_res = zeros(n_sims, length(ms));
    xtls_res = zeros(n_sims, length(ms));
    xmle_res = zeros(n_sims, length(ms));

    fprintf('For %i %s simulations for %i column matrices \n', n_sims, model_names{model}, N); 
    for i = 1:n_sims
        % set seed for each simulation
        rng(i+500);

        % generate true solution from which data is generated
        u = rand(N, 1);
        xtrue = tan(pi*(u-0.5));
        %xtrue = randn(N,1);%-0.5;
        %xtrue = xtrue/norm(xtrue);

        sigma = 0.1;
        switch model
            case 1 % exponential clipping
                %sigma = 1;    % standard deviation of noise on RHS
                lambda = 2;     % exponential of rate lambda has mean 1/lambda
                gamma = 2;      % smaller means clipped occurs sooner                
                U = rand(M,N);
                G = -(1/lambda)*log(1 - U).*sign(randn(M,N));
                H = sign(G).*min(abs(G),gamma);
            case 2 % gaussian
                %sigma = 1;    % standard deviation of noise on RHS
                rho = 2;        % standard deviation of Gaussian noise for design matrix
                H = 10*(randn(M,N));
                G = H + rho*randn(M,N);
            case 3 % rounding
                %sigma = 1;
                rtd = 0;            % round to digit, 0 is ones spot, 1 is tenths, etc
                c = 0.5*10^(-rtd);  % uncertainty parameter, i.e. g ~ unif(h - c, h + c)
                G = 10*(rand(M,N)); 
                H = round(G,rtd);
            case 4 % floating point
                %sigma = 1;
                sig_figs = 2;       % number of significant figures to use
                dyn_range = 3;      % large dyn_range increases variability of exponents for design                
                G = randn(M,N).*10.^(randi([0,dyn_range], M,N));
                D = floor(log10(abs(G)));
                H = round(G./ 10.^D, sig_figs-1).* 10.^D;
                D = 5*10.^(D-sig_figs);
                D_array{i} = D;
        end

        % generate measured data
        ytrue = G*xtrue;
        eta = sigma*randn(M,1);
        y = ytrue + eta;

        % store data to be used again
        rhs_array{i} = y;
        G_array{i} = G;
        H_array{i} = H;
        true_solution_array{i} = xtrue;
    end


    for jj = 1:length(ms) 
        tic
        m = ms(jj);
        if opts.Verbose
            fprintf('\n \n \n %s:  number of rows %i \n', model_names{model}, ms(jj))
            fprintf('\t || \t AMLE Error \t || \t OLS Error \t || \t TLS Error \n')
            fprintf('============================================================================ \n')
        end
        
        % main loop, each ii is new instance of experiment
        for ii = 1:n_sims     
            % retrieve desired simulation run and take desired number of rows
            xtrue = true_solution_array{ii}(1:N); 
            y = rhs_array{ii}(1:m);
            H = H_array{ii}(1:m,:);
            G = G_array{ii}(1:m,:);

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
                    D = D_array{ii}(1:m,:);
                    neg_L =@(z) float_neg_loglikelihood(y, H, z, D, sigma);
            end

            % initialize to OLS solution
            %x0 = randn(N,1);
            x0 = xols;
            xmle = minFunc(neg_L,x0,opts);

            % store errors
            xols_error(ii, jj) = norm(xols  - xtrue)/norm(xtrue);
            xtls_error(ii, jj) = norm(xtls - xtrue)/norm(xtrue);
            xmle_error(ii, jj) = norm(xmle - xtrue)/norm(xtrue);
            
            
            %store residuals
            xols_res(ii, jj) = norm(G*xols - y);
            xtls_res(ii, jj) = norm(G*xtls - y);
            xmle_res(ii, jj) = norm(G*xmle - y);
            if opts.Verbose
                fprintf('Sim %i \t || \t %4.6f \t || \t %4.6f \t || \t %4.6f \n', ...
                    ii, norm(xmle  - xtrue), norm(xols  - xtrue), norm(xtls - xtrue))      
            end
        end
        toc 
    end
    
    save_string = strcat('data/',model_names{model}, '.mat');
    save(save_string)
    
    median_error_array(:, 1, model) = median(xmle_error);
    median_error_array(:, 2, model) = median(xols_error);
    median_error_array(:, 3, model) = median(xtls_error);
    
    mean_error_array(:, 1, model) = median(xmle_error);
    mean_error_array(:, 2, model) = median(xols_error);
    mean_error_array(:, 3, model) = median(xtls_error);
    
end

save('data/error_array.mat', 'mean_error_array', 'median_error_array')

%%

figure()
sgtitle(strcat('Mean relative error vs. row/column ratio'),  'Interpreter','latex', 'FontSize',18)
for model = 1:4
    % remove the following before pushing to github
    subplot(2,2,model)
    co = colorOrder( 'highcontrast');
    set(groot,'defaultAxesColorOrder',co); set(gca,'defaultAxesColorOrder',co); set( gca, 'ColorOrder', co );
    
    abc = loglog(ms/N, squeeze(median_error_array(:,:,model)), 'Marker', 's', 'LineWidth', 3);
    ax = gca;
    set(ax, 'FontSize', 12)
    title(model_names{model}, 'Interpreter','latex', 'FontSize',16);
    if model == 1 || model == 3
        ylabel('Median rel. error','Interpreter','latex', 'FontSize',16);
    end
    if model == 3 || model == 4 %[3,4]
        xlabel('$m/n$', 'Interpreter','latex', 'FontSize',16);
    end
    if model == 2
        legend({'AML (proposed)','OLS','TLS'}, 'Interpreter','latex', 'FontSize',12);
        legend boxoff
    end
    abc(1).LineStyle = '-'; abc(1).LineWidth = 3; abc(1).Marker = 'o'; abc(1).MarkerSize = 10;
    abc(2).LineStyle = '--'; abc(2).LineWidth = 3; abc(2).Marker = 'x'; abc(2).MarkerSize = 10;
    abc(3).LineStyle = ':'; abc(3).LineWidth = 3; abc(3).Marker = '+'; abc(3).MarkerSize = 10;
    grid on
end

set(gcf, 'color', 'w')
