FILENAME = 'C:\Users\n01388138\Downloads\FCNN_3mod_run5_200dec.txt';
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
M = movmean(rwd,[100 0]);%moving mean of last 100 epiosdes
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
episode_count = 400;
rwd= rwd(1:episode_count,1);
X=1:episode_count;
%scatter(X,rwd);
plot(rwd,'Linewidth', 0.1);
hold all;
M = movmean(rwd,[50 0]);%moving mean of last 100 epiosdes
plot(M, 'Linewidth', 2);
hold off;
xlabel('Episodes','FontSize',14);
ylabel('Reward','FontSize',14);
legend('Episodic Reward','Moving Average (50 episodes)','FontSize',12, 'Location','best')
max_x = (numel(rwd)+100);
xlim([-100 max_x])
filename = strcat('movemean_rwd.png');
saveas(gcf,filename);

%% exploration vs. exploitation
eps = expl * 100;
exp = 100 - eps;
e = [1:numel(eps)];
yyaxis left
plot(e,eps);
ylabel('Exploration')
ylim([-10 110])

yyaxis right
plot(e,exp);
ylabel('Exploitation');
ylim([-10 110])
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