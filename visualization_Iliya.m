FILENAME = 'C:\Users\n01388138\Downloads\CNN_actoins_test1.txt';
[rwd, loss] = dataImport_iliya(FILENAME);
%% average reward per episode
%rwd = MODULES1100Ep.r;
%rwd = rwd(1000:end,1);
figure();
plot(rwd)
rwd = cumsum(rwd);
idx = 1:numel(rwd);
idx = idx';
avgr = rwd ./ idx;
M = movmean(avgr,[100 0]);%moving mean of last 100 epiosdes
plot(avgr)
hold all
plot(M)
hold off
xlabel('Episodes','FontSize',14);
ylabel('Average Reward','FontSize',14);
xlim([-100 5100])
filename = strcat('avg_rwd.png');
%saveas(gcf,filename);
% 

%% reward plot (moving mean)
[rwd, loss] = dataImport_iliya(FILENAME);
M = movmean(rwd,[100 0]);%moving mean of last 100 epiosdes
plot(M)
xlabel('Episodes','FontSize',14);
ylabel('Reward','FontSize',14);
legend('Moving Average (100 episodes)','FontSize',12, 'Location','best')
xlim([-100 5100])
filename = strcat('movmean_rwd.png');
saveas(gcf,filename);


%% exploration vs. exploitation
eps = run1.eps * 100;
exp = 100 - eps;
e = [1:4999];
yyaxis left
plot(e,eps);
ylabel('Exploration')

yyaxis right
plot(e,exp);
ylabel('Exploitation');
xlabel('episodes');
title('Exploration vs. Exploitation');


%% loss per episode
figure();
%loss = None;
%loss = cumsum(loss);
%idx = 1:numel(loss);
%idx = idx';
%loss = loss ./ idx;
plot(loss(32:end))
xlabel('Episodes','FontSize',14);
ylabel('Loss','FontSize',14);
xlim([-100 5100])
filename = strcat('loss.png');
saveas(gcf,filename);
