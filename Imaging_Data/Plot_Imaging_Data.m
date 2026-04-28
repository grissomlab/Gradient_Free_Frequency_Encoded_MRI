
clc;
clear all;
close all;

load('Imaging_Data.mat')

%% magnitude images 

b0ax = [ 0 1.3e8];
b0axinv= [ 0 1.1e8];
b0axmow= [ 0 1.4e8];
bsax = [ 1e8 1e9];
bsaxmow = [0.8e8 1.4e9];


figure(100); t = tiledlayout(3,2,'TileSpacing','compact'); t.Padding = 'compact';

nexttile 
imagesc(flipud(abs(paddata_B0_MO_crop))); axis square; colormap gray; ...
      xticks ([]), yticks([]); caxis (b0ax);
title('B0 Frequency Encoded', 'FontSize', 20, 'FontName', 'Verdana');
ylabel('Mineral Oil', 'FontSize', 24, 'FontName', 'Verdana');

nexttile 
imagesc((abs(paddata_BS_MO_crop))); axis square; colormap gray; ...
    xticks ([]), yticks([]); caxis (bsax);
title('BS Frequency Encoded', 'FontSize', 20, 'FontName', 'Verdana');

nexttile 
imagesc((abs(paddata_B0_MOW_crop))); axis square; colormap gray; ...
    ylabel({'Mineral Oil','and Water'}, 'FontSize', 24, 'FontName', 'Verdana');...
     xticks ([]), yticks([]);caxis (b0axmow);

nexttile 
imagesc((abs(paddata_BS_MOW_crop))); axis square; colormap gray; ...
   xticks ([]), yticks([]); caxis (bsaxmow); 


nexttile 
imagesc((abs(paddata_B0_inv_MOW_crop))); axis square; colormap gray; ...
    xticks ([]), yticks([]); caxis (b0axinv);
ylabel({'Mineral Oil','and Water','- IR Sequence', }, 'FontSize', 24, 'FontName', 'Verdana');


nexttile 
imagesc((abs(paddata_BS_inv_MOW_crop))); axis square; colormap gray; ...
    xticks ([]), yticks([]); caxis (bsaxmow);


%% 1D plots 

hline = 16;
yy_mm = linspace ( 0, 7,55);

figure(200); t = tiledlayout(3,1,'TileSpacing','compact'); 
t.Padding = 'compact';

nexttile 
plot1d_tiled(yy_mm,paddata_B0_MO_crop,paddata_BS_MO_crop, hline);
camroll(270)
set(gca, 'XTick', []);
set(gca, 'YTick', []);

nexttile 
plot1d_tiled(yy_mm,paddata_B0_MOW_crop,paddata_BS_MOW_crop, hline);
camroll(270)
set(gca, 'XTick', []);
set(gca, 'YTick', []);

nexttile 

plot1d_tiled(yy_mm,paddata_B0_inv_MOW_crop,paddata_BS_inv_MOW_crop, hline);
camroll(270)
legend('\color{red}B0 encoded', '\color{green}BS encoded','Orientation','horizontal','FontSize',35, 'FontName','Verdana'); 
set(gca, 'XTick', []);
set(gca, 'YTick', []);

%% Phase image

figure(300); 
t = tiledlayout(1,2,'TileSpacing','compact'); 
t.Padding = 'compact';

% --- Left column (3 tiles + colorbar) ---
tLeft = tiledlayout(t, 3,1,'TileSpacing','compact');
tLeft.Padding = 'compact';
tLeft.Layout.Tile = 1;

nexttile(tLeft)
imagesc(flipud(angle(paddata_B0_MO_crop))); axis square; colormap gray; xticks([]), yticks([]);
nexttile(tLeft)
imagesc((angle(paddata_B0_MOW_crop))); axis square; colormap gray; xticks([]), yticks([]);
nexttile(tLeft)
imagesc((angle(paddata_B0_inv_MOW_crop))); axis square; colormap gray; xticks([]), yticks([]);

cbLeft = colorbar;
cbLeft.Layout.Tile = 'east';
cbLeft.TickLabels = {};
cbLeft.Position(3) = 0.05; % wider bar

clim_vals = [-pi, pi];
for ax = findobj(tLeft.Children, 'Type', 'Axes')'
    clim(ax, clim_vals);
end

% --- Right column (3 tiles + colorbar) ---
tRight = tiledlayout(t, 3,1,'TileSpacing','compact');
tRight.Padding = 'compact';
tRight.Layout.Tile = 2;

nexttile(tRight)
imagesc((angle(paddata_BS_MO_crop))); axis square; colormap gray; xticks([]), yticks([]);
nexttile(tRight)
imagesc((angle(paddata_BS_MOW_crop))); axis square; colormap gray; xticks([]), yticks([]);
nexttile(tRight)
imagesc((angle(paddata_BS_inv_MOW_crop))); axis square; colormap gray; xticks([]), yticks([]);

cbRight = colorbar;
cbRight.Layout.Tile = 'east';
cbRight.TickLabels = {};
cbRight.Position(3) = 0.05; % wider bar

for ax = findobj(tRight.Children, 'Type', 'Axes')'
    clim(ax, clim_vals);
end

set(gcf, 'Color', 'white');
set(findobj(gcf, 'Type', 'Axes'), 'Color', 'white');
