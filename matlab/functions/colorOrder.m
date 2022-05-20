function c = colorOrder( colorName )
% co = colorOrder( nameOfColorScheme )
%   returns types of color schemes for plotting,
%   which can be used like:
%> set(groot,'defaultAxesColorOrder',co)
%> set(gca,'defaultAxesColorOrder',co);
%> set( gca, 'ColorOrder', co ) (won't affect existing colors though)
% Ex: plot( [ 1:7; 1+(1:7)], 'linewidth',2 )
%
% Available schmes:
%   'newmatlab' (default in R2014b and later)
%   'oldmatlab' (default before R2014b)
%   ... and these schemes by Paul Tol (see his tech report: 
% https://personal.sron.nl/~pault/data/colourschemes.pdf )
%   'bright' "qualitative colour scheme that is colour-blind safe. The main
%       scheme for lines and their labels."
%       blue, cyan, green, yellow, red, purple, grey
%   'highcontrast' "qualitative colour scheme, an alternative to the bright scheme of Fig. 1 that is colourblind safe and optimized for contrast. The samples underneath are shades of grey with the same luminance;
%       this scheme also works well for people with monochrome vision and in a monochrome printout."
%       white, yellow, red, blue, black
%   'vibrant' "qualitative colour scheme, an alternative to the bright scheme of Fig. 1 that is equally colourblind safe. It has been designed for data visualization framework TensorBoard, built around their signature
%       orange FF7043. That colour has been replaced here to make it
%       print-friendly."
%       blue, cyan, teal, orange, red, magenta, grey

% set(groot,'defaultAxesColorOrder',co)
% ax = gca;
% ax.ColorOrderIndex = 1;

switch lower(colorName)
    
    case 'newmatlab' % see https://www.mathworks.com/help/matlab/graphics_transition/why-are-plot-lines-different-colors.html#buhi77f-2
        c = [0    0.4470    0.7410
        0.8500    0.3250    0.0980
        0.9290    0.6940    0.1250
        0.4940    0.1840    0.5560
        0.4660    0.6740    0.1880
        0.3010    0.7450    0.9330
        0.6350    0.0780    0.1840];
    case 'oldmatlab'
        c = [0         0    1.0000
        0    0.5000         0
        1.0000         0         0
        0    0.7500    0.7500
        0.7500         0    0.7500
        0.7500    0.7500         0
        0.2500    0.2500    0.2500];
    case 'bright'
        c = [68,119,170
        102,204,238
        34,136,51
        204,187,68
        238,102,119
        170,51,119
        187,187,187]/255;
    case 'highcontrast'
        c = [0,0,0
        221,170,51
        187,85,102
        0,68,136
        187,187,187]/255;
    case 'vibrant'
        c=[0,119,187
        51,187,238
        0,153,136
        238,119,51
        204,51,17
        238,51,119
        187,187,187]/255;
    otherwise
        error('bad value');
end