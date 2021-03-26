if ~exist('median_error_array','var')
    load('data/vary_rows_error_array.mat')
end

use_avg = 'Median';

if strcmp(use_avg, 'Median')
    err = median_error_array;
else
    if strcmp(use_avg, 'Mean')
        err = mean_error_array;
    else
        fprintf('Select mean or median for error calculuations')
    end
end

figure()
sgtitle(strcat(use_avg, ' relative error vs. row/column ratio'),  'Interpreter','latex', 'FontSize',18)
for model = 1:4
    if exist('colorOrder')
        subplot(2,2,model)
        co = colorOrder( 'highcontrast');
        set(groot,'defaultAxesColorOrder',co); set(gca,'defaultAxesColorOrder',co); set( gca, 'ColorOrder', co );
    end

    abc = loglog(ms/N, squeeze(err(:,:,model)), 'Marker', 's', 'LineWidth', 3);
    ax = gca;
    set(ax, 'FontSize', 12)
    title(model_names{model}, 'Interpreter','latex', 'FontSize',16);
    if model == 1 || model == 3
        ylabel('Rel. error','Interpreter','latex', 'FontSize',16);
    end
    if model == 3 || model == 4 %[3,4]
        xlabel('$m/n$', 'Interpreter','latex', 'FontSize',16);
    end
    if model == 2
        legend({'AMLE (proposed)','OLS','TLS'}, 'Interpreter','latex', 'FontSize',12);
        legend boxoff
    end
    abc(1).LineStyle = '-'; abc(1).LineWidth = 3; abc(1).Marker = 'o'; abc(1).MarkerSize = 10;
    abc(2).LineStyle = '--'; abc(2).LineWidth = 3; abc(2).Marker = 'x'; abc(2).MarkerSize = 10;
    abc(3).LineStyle = ':'; abc(3).LineWidth = 3; abc(3).Marker = '+'; abc(3).MarkerSize = 10;
    grid on
end

set(gcf, 'color', 'w')

