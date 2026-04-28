% S21 Insertion Loss Plot - Injection Transformer
clear all;

% Load data
T = readmatrix('Transformer_s21.csv');
freq = T(1:102, 1) * 1e-6;  % Convert to MHz
s21 = T(1:102, 4);

% Plot
figure;
plot(freq, s21, 'LineWidth', 4, 'Color', 'k');
hold on;
xline(2.075, '--', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);

% Formatting
ylim([-6 1]);
xlim([1.975 2.175]);
set(gca, 'FontSize', 20, 'FontName', 'Verdana');
set(gcf, 'Color', [1 1 1]);
set(gca, 'Color', [1 1 1]);
set(gca, 'Box', 'off');

% Labels
xlabel('Frequency (MHz)');
ylabel('Magnitude (dB)');
title({'Injection Transformer Insertion Loss (S_{21}) '});

% Add text annotation for Larmor frequency
text(2.075, 5, 'Larmor frequency', 'HorizontalAlignment', 'center', ...
     'FontSize', 12, 'FontName', 'Verdana');