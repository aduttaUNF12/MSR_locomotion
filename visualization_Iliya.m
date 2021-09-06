FILENAME = 'C:\Users\n01388138\Downloads\5000_first_vector_implementation_test.txt';
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
plot(avgr)
xlabel('Episodes','FontSize',14);
ylabel('Average Reward','FontSize',14);
xlim([-100 5100])
filename = strcat('avg_rwd.png');
saveas(gcf,filename);
% 

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