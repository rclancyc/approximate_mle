if ~exist('err_array', 'var')
    load('data/data_for_histograms.mat')
end


figure();
if exist('colorOrder')
    co = colorOrder( 'highcontrast');
    set( gca, 'ColorOrder', co );
end
sgtitle('Box plots for relative error',  'Interpreter','latex', 'FontSize',16)
for model = 1:4
    subplot(2,2,model);
    set(gca,'FontSize',12)
    boxplot(err_array(:,1:3,model), {'AML (proposed)', 'OLS', 'TLS'})
    title(model_names{model}, 'Interpreter', 'latex','FontSize',16);
    ax = gca;
    if model == 1 || model == 3
        ylabel('Rel. error','Interpreter','latex', 'FontSize',16);
    end

    ax.YAxis.Scale ="log";
    ax.YLim = [1e-3 1e1];
    grid on
end
set(gcf,'color','w')




%%
figure();

newLimHi = 0;
chart_names = {'Exp clip', 'Gaussian', 'Rounding', 'Floating'};
sgtitle('Histogram of error ratios', 'Interpreter','latex', 'FontSize', 16)
for model = 1:4
    mle_ols = err_array(:,1,model)./err_array(:,2,model);
    mle_tls = err_array(:,1,model)./err_array(:,3,model);

    minbin = min(min(mle_ols), min(mle_tls));
    maxbin = max(max(mle_ols), max(mle_tls));
    nbins = 75;
    bins = logspace(-2, 1, nbins);
    subplot(4,2,2*(model-1)+1)
    h1 = histogram(mle_ols, bins,'Normalization', 'count');
    set(gca, 'Xscale', 'log','FontSize',12, 'YTickLabel',[]);
    xlim([1e-2 1e1])
    ylabel(chart_names{model}, 'Interpreter','latex', 'FontSize', 16);
    grid on
    if 2*(model-1)+1 == 1
        title('AML over OLS', 'Interpreter', 'latex', 'FontSize', 14)
    end
    if 2*(model-1)+1 ~=7
        set(gca, 'XTickLabel',[]);
    end
    hold on;

    subplot(4,2,2*model)
    h2 = histogram(mle_tls,bins,'Normalization', 'count');
    set(gca, 'Xscale', 'log','FontSize',11, 'YTickLabel',[]);
    
    xlim([1e-2 1e1])
    newLimLo = min(h1.Parent.YLim(1),h2.Parent.YLim(1));
    newLimHi = max([h1.Parent.YLim(2),h2.Parent.YLim(2), newLimHi]);
    grid on
    h1.Parent.YLim = [newLimLo newLimHi];
    h2.Parent.YLim = [newLimLo newLimHi];
    
    if 2*model == 2
        title('AML over TLS',  'Interpreter', 'latex', 'FontSize', 14)
    end
    if 2*model ~=8
        set(gca, 'XTickLabel',[]);
    end
    hold on;
end

for k = 1:4
    subplot(4,2,2*(k-1)+1); ylim([newLimLo 1500]); 
    plot([1, 1], [0, 1500], 'LineWidth',1.5, 'color', 'red')
    subplot(4,2,2*k); ylim([newLimLo 1000]); 
    plot([1 1], [0 1500],'LineWidth',1.5, 'color', 'red')    
end

set(gcf,'color','w')

