addpath(genpath('functions'))

if ~exist('mean_error_array','var')
    load('data/vary_cols_error_array.mat')
end 
%use_avg = 'Mean';
use_avg = 'Median';

if strcmp(use_avg, 'Median')
    err = median_error_array;
    err_ratio = median_error_ratio;
else
    if strcmp(use_avg, 'Mean')
        err = mean_error_array;
        err_ratio = mean_error_ratio;
    else
        fprintf('Select mean or median for error calculuations')
    end
end

figure()
sgtitle(strcat(use_avg,' relative error vs. column count'),  'Interpreter','latex', 'FontSize',18)
for model = 1:4
    subplot(2,2,model)
    if exist('colorOrder')
        co = colorOrder( 'highcontrast');
        set(groot,'defaultAxesColorOrder',co); set(gca,'defaultAxesColorOrder',co); set( gca, 'ColorOrder', co );
    end
    
    abc = loglog(ns, squeeze(err(:,:,model)), 'Marker', 's', 'LineWidth', 3);
    ax = gca;
    set(ax, 'FontSize', 12)
    title(model_names{model}, 'Interpreter','latex', 'FontSize',16);
    if model == 1 || model == 3
        ylabel('Rel. error','Interpreter','latex', 'FontSize',16);
    end
    if model == 3 || model == 4 %[3,4]
        xlabel('$n$', 'Interpreter','latex', 'FontSize',16);
    end
    if model == 2
        legend({'AML (proposed)','OLS','TLS'}, 'Interpreter','latex', ...
            'FontSize',12, 'Location', 'northwest');
        legend boxoff
    end
    abc(1).LineStyle = '-'; abc(1).LineWidth = 3; abc(1).Marker = 'o'; abc(1).MarkerSize = 10;
    abc(2).LineStyle = '--'; abc(2).LineWidth = 3; abc(2).Marker = 'x'; abc(2).MarkerSize = 10;
    abc(3).LineStyle = ':'; abc(3).LineWidth = 3; abc(3).Marker = '+'; abc(3).MarkerSize = 10;
    grid on
end
clrz = get(abc, 'Color');
c2 = clrz{2};
c3 = clrz{3};
set(gcf, 'color', 'w')

