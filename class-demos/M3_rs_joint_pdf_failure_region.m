%% rs_joint_pdf_failure_region.m
% CV2090 | The R-S reliability problem: the picture behind slide 2
%
%   left  panel : contours of f_RS(r,s), the line g = 0, failure side shaded
%   right panel : the same f_RS as a surface, failure side in red
%                 (the volume under the red part IS p_f)
%
%   g(R,S) = R - S,        failure = {g <= 0} = {S >= R}
%
% R and S jointly normal. 
%
% beta and p_f printed to the Command Window. 

clear; close all; 
clc;

%% 1. inputs -- change these
% 'figure' exaggerates the overlap so the red volume is visible on a slide.
% 'figure_with_rho' exaggerated case with rho = 0.5
% 'caseA'  is the realistic pair used in the Python demo. At beta = 2.19 the
%          failure lobe is a flat sliver: true to life, useless as a picture.
preset = 'figure';

switch preset
    case 'figure'
        muR = 200;  sigR = 30;      % capacity R
        muS = 165;  sigS = 30;      % demand   S
        rho = 0.0;                  % correlation, in (-1, 1)
    case 'figure_with_rho'
        muR = 200;  sigR = 30;      % capacity R
        muS = 165;  sigS = 30;      % demand   S
        rho = 0.5;                  % correlation, in (-1, 1)
    case 'caseA'
        muR = 200;  sigR = 20;
        muS = 130;  sigS = 25;
        rho = 0.0;
end

showLabels = true;      % false -> no 'Fail' / 'Safe' / 'g = 0' text
nGrid      = 201;       % grid for the contours and the surface

% standard normal CDF, written out so no toolbox is needed
Phi = @(z) 0.5 * erfc(-z / sqrt(2));

%% 2. the answer two ways, printed for checking
% (a) exact, from the safety margin Z = R - S                      
muZ  = muR - muS;
sigZ = sqrt(sigR^2 - 2*rho*sigR*sigS + sigS^2);
beta = muZ / sigZ;
pf_exact = Phi(-beta);

% (b) by conditioning on R:  p_f = int [1 - F_S|R(r|r)] f_R(r) dr  
rInt   = linspace(muR - 6*sigR, muR + 6*sigR, 2001);
fR     = exp(-0.5*((rInt - muR)/sigR).^2) / (sigR*sqrt(2*pi));
muScon = muS + rho*(sigS/sigR)*(rInt - muR);    % mean of S given R = r
sgScon = sigS * sqrt(1 - rho^2);                % s.d.  of S given R = r
pf_cond = trapz(rInt, (1 - Phi((rInt - muScon)/sgScon)) .* fR);

fprintf('preset                    = %s\n', preset);
fprintf('beta                      = %.4f\n', beta);
fprintf('p_f  exact, Phi(-beta)    = %.6f\n', pf_exact);
fprintf('p_f  by conditioning on R = %.6f\n', pf_cond);

%% 3. grid over the r-s plane
% Same range on both axes, so the g = 0 line really is at 45 degrees
lo = min(muR - 4*sigR, muS - 4*sigS);
hi = max(muR + 4*sigR, muS + 4*sigS);

v = linspace(lo, hi, nGrid);
[Rg, Sg] = meshgrid(v, v);

%% 4. joint pdf of R and S, written out (no mvnpdf)
zR = (Rg - muR) / sigR;
zS = (Sg - muS) / sigS;
qq = (zR.^2 - 2*rho*zR.*zS + zS.^2) / (1 - rho^2);
f  = exp(-qq/2) / (2*pi*sigR*sigS*sqrt(1 - rho^2));

%% 5. limit state function and the failure region
g    = Rg - Sg;                 % g <= 0 is failure
fail = (g <= 0);

%% 6. left panel: contours, the g = 0 line, failure side shaded
figure('Color', 'w', 'Position', [80 80 1150 480]);

subplot(1,2,1); hold on; box on;

% the failure region is the triangle above the diagonal
patch([lo hi lo], [lo hi hi], [1 0.82 0.82], 'EdgeColor', 'none');

% contours of f_RS at 1, 2 and 3 standard deviations
fPeak  = 1 / (2*pi*sigR*sigS*sqrt(1 - rho^2));
levels = fPeak * exp(-[3 2 1].^2 / 2);          % must be ascending
contour(Rg, Sg, f, levels, 'b', 'LineWidth', 1.5);

plot([lo hi], [lo hi], 'r-', 'LineWidth', 2);   % the line g = 0
plot(muR, muS, 'k+', 'MarkerSize', 12, 'LineWidth', 1.5);   % the mean point

if showLabels
    text(lo + 0.12*(hi-lo), lo + 0.84*(hi-lo), 'Fail', 'FontSize', 16);
    text(lo + 0.76*(hi-lo), lo + 0.16*(hi-lo), 'Safe', 'FontSize', 16);
    text(lo + 0.70*(hi-lo), lo + 0.76*(hi-lo), 'g(r,s) = 0', ...
         'FontSize', 13, 'Color', 'r', 'Rotation', 45);
end

axis([lo hi lo hi]); axis square;
xlabel('r'); ylabel('s');
set(gca, 'FontSize', 14);

%% 7. right panel: the same pdf as a surface, failure side in red
subplot(1,2,2); hold on;

fSafe = f;  fSafe(fail)  = NaN;         % safe side
fFail = f;  fFail(~fail) = NaN;         % failure side

surf(Rg, Sg, fSafe, 'EdgeColor', 'none', 'FaceColor', [0.55 0.65 0.95]);
surf(Rg, Sg, fFail, 'EdgeColor', 'none', 'FaceColor', [0.90 0.30 0.30]);

plot3([lo hi], [lo hi], [0 0], 'k-', 'LineWidth', 1.5);     % g = 0 on the floor

xlim([lo hi]); ylim([lo hi]);
xlabel('r'); ylabel('s'); zlabel('f_{RS}(r,s)');

view(140, 30);    % the red lobe should face you.
                  % If it ends up hidden behind the blue peak, use view(-40, 30).

grid on; box on;
camlight; lighting gouraud; material dull;
set(gca, 'FontSize', 14);

%% 8. to save the figure for the slides, uncomment 
% exportgraphics(gcf, 'rs_failure_region.png', 'Resolution', 300);
