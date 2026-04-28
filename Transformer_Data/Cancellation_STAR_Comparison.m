close all;
clear all;
clc;

load('Cancellation_Data.mat')

colors = {[0 0.4470 0.7410], [0.6350 0.0780 0.1840],[0.4660 0.6740 0.1880], [0.9290 0.6940 0.1250], [0.4940 0.1840 0.5560]}; % blue, maroon, green, yellow, purple


data_ch1log=20*log10(data_ch1);
data_ch2log=20*log10(data_ch2);
data_ch3log=20*log10(data_ch3);
data_ch4log=20*log10(data_ch4);
data_ch5log= 20*log10(data_ch5);

corr = data_ch2(25:end,:);
uncorr = data_ch1(25:end,:);
Canc_preemp = ((mean(abs(corr(:)))-mean(abs(uncorr(:))))./mean(abs(uncorr(:)))).*100;

figure; 
t = 1:1:488;
a = plot(t, abs(data_ch4(25:end)), t, abs(data_ch5(25:end)) ,'LineWidth',3);

[a(1).Color] = colors{1,1}; % blue
[a(2).Color] = [0.3010 0.7450 0.9330]; % light blue

legend ('Leaked Transmit Signal', 'Leaked Transmit Signal with Pre-emphasis');
set(legend, 'FontSize',20, 'FontName','Verdana');
set (gca, 'FontSize',20, 'FontName','Verdana' );
xlim( [0 488])
set(gca, 'XTick', []);
set(gca,'FontSize',20,'fontname','Verdana'); 
set(gca, 'Box', 'off')
legend('boxoff')
figure; 
t = 1:1:488;
a = plot(t, abs(data_ch1(25:end)), t, abs(data_ch2(25:end)) ,t, abs(data_ch3log(25:end)), 'LineWidth',3);

[a(1).Color] = colors{1,3}; % green 
[a(2).Color] = [0 0.49 0.2]; % light green
[a(3).Color] = [0 0 0]; % black
legend ('Signal Post-Cancellation', 'Signal Post-Cancellation with Pre-emphasis', 'Receive channel Noise Floor');
set(legend, 'FontSize',20, 'FontName','Verdana');
set (gca, 'FontSize',20, 'FontName','Verdana' );
% axis tight;
ylim ([0 900]);
xticks([1 243 486])
xticklabels ({' 0', '10.24' ,'20.48'});
yticks([1 450 900])
xlim( [0 488])
set(gca, 'Box', 'off')
legend('boxoff')
set(gca,'FontSize',20,'fontname','Verdana'); 


%% figure 2c
figure; 
t = 1:1:488;
a = plot(t, abs(data_ch5log(25:end)), t, abs(data_ch2log(25:end)), t, abs(data_ch3log(25:end)) ,'LineWidth',3);

[a(1).Color] = colors{1,1}; % blue
[a(2).Color] = colors{1,3}; % green
[a(3).Color] = [0 0 0]; % black
legend ('Leaked Transmit Signal' , 'Post-Cancellation Residual', 'Receive channel Noise Floor');
set(legend, 'FontSize',16, 'FontName','Verdana');
set (gca, 'FontSize',16, 'FontName','Verdana' );
% axis tight;
ylim ([0 110]);
xticks([1 243 486])
xticklabels ({' -10.24', '0' ,'10.24'});
xlim( [0 488])
set(gca,'FontSize',16,'fontname','Verdana'); 
set(gca, 'Box', 'off')
legend('boxoff')

fclose('all');

