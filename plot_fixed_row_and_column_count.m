if ~exist('err_array', 'var')
    load('data/data_for_histograms.mat')
end


figure()
if exist('colorOrder')
    co = colorOrder( 'highcontrast');
    %set(groot,'defaultAxesColorOrder',co); 
    %set(gca,'defaultAxesColorOrder',co); 
    set( gca, 'ColorOrder', co );
end
%sgtitle('Box plots for relative error $\frac{\|\mathbf{x}_{EST} - \mathbf{x}_{TRU}\|}{\|\mathbf{x}_{TRU}\|}$',  'Interpreter','latex', 'FontSize',16)
sgtitle('Box plots for relative error',  'Interpreter','latex', 'FontSize',16)
for model = 1:4
    subplot(2,2,model);
    set(gca,'FontSize',12)
    boxplot(err_array(:,1:3,model), {'AMLE (proposed)', 'OLS', 'TLS'})%, 'FontSize',16); %'Interpreter','latex','
    %title('Box plot for relative error $\left(\frac{\|\mathbf{x}_{EST} - \mathbf{x}_{TRU}\|}{\|\mathbf{x}_{TRU}\|}\right)$', ... 
    %    'Interpreter', 'latex','FontSize',18);
    title(model_names{model}, 'Interpreter', 'latex','FontSize',16);
    ax = gca;
    if model == 1 || model == 3
        ylabel('Rel. error','Interpreter','latex', 'FontSize',16);
    end

    ax.YAxis.Scale ="log";
    %ax.YLim = [1e-3 1e1];
    grid on
end
set(gcf,'color','w')








%%
figure();

chart_names = {'Exp clip', 'Gaussian', 'Rounding', 'Floating'};
sgtitle('Histogram of error ratios', 'Interpreter','latex', 'FontSize', 16)
for model = 1:4
    mle_ols = err_array(:,1,model)./err_array(:,2,model);
    mle_tls = err_array(:,1,model)./err_array(:,3,model);

    minbin = min(min(mle_ols), min(mle_tls));
    maxbin = max(max(mle_ols), max(mle_tls));
    nbins = 75;
    bins = logspace(-1, log10(10), nbins);
    subplot(4,2,2*(model-1)+1)
    h1 = histogram(mle_ols, bins,'Normalization', 'count');
    set(gca, 'Xscale', 'log','FontSize',12, 'YTickLabel',[]);
    xlim([1e-1 10])
    ylabel(chart_names{model}, 'Interpreter','latex', 'FontSize', 16);
    grid on
    if 2*(model-1)+1 == 1
        title('AMLE over OLS', 'Interpreter', 'latex', 'FontSize', 14)
    end
    if 2*(model-1)+1 ~=7
        set(gca, 'XTickLabel',[]);
    end

    subplot(4,2,2*model)
    h2 = histogram(mle_tls,bins,'Normalization', 'count');
    set(gca, 'Xscale', 'log','FontSize',12, 'YTickLabel',[]);
    
    xlim([1e-1 10])
    newLimLo = min(h1.Parent.YLim(1),h2.Parent.YLim(1));
    newLimHi = max(h1.Parent.YLim(2),h2.Parent.YLim(2));
    grid on
    h1.Parent.YLim = [newLimLo newLimHi];
    h2.Parent.YLim = [newLimLo newLimHi];
    
    if 2*model == 2
        title('AMLE over TLS',  'Interpreter', 'latex', 'FontSize', 14)
    end
    if 2*model ~=8
        set(gca, 'XTickLabel',[]);
    end

end

set(gcf,'color','w')




