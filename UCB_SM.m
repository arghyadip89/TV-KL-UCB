n_trials=100000;
n_arms=5;% number of arms
iteration_total=100;
%Total number of times an arm is played in an iteration
played_iteration=zeros(iteration_total,n_arms);
trial=zeros(1,n_trials);
regret=zeros(iteration_total,n_trials);% Regret for each trial
regret_1og=zeros(iteration_total,n_trials);% Regret for each trial

for ii=1:n_trials
    trial(ii)=ii;
end

for iteration=1:iteration_total
	mu_P=zeros(1,n_arms);% Transition prob from state 0 to 1
	mu_Q=zeros(1,n_arms);% Transition prob from state 1 to 0
	pai=zeros(1,n_arms);% Steady state st prob of state 1 (mean)
	current_state=zeros(1,n_arms);% Running counter for current state
	mu_est=zeros(1,n_arms);% Estimated mean for each arm
	played=zeros(1,n_arms);% number of times each arm picked
	UCB_est=zeros(1,n_arms);% Estimation of UCB for eac arm
	%regret=zeros(1,n_trials);% Regret for each trial

%for ii=1:n_trials
%    trial(ii)=ii;
%end

mu_P(5)=0.1;
mu_P(4)=0.2;
mu_P(3)=0.3;
mu_P(2)=0.4;
mu_P(1)=0.5;
%mu_P(6)=0.22;
%mu_P(7)=0.38;
%mu_P(8)=0.45;
%mu_P(9)=0.6;
%mu_P(10)=0.7;

mu_Q(1)=0.4;
mu_Q(2)=0.55;
mu_Q(3)=0.65;
mu_Q(4)=0.65;
mu_Q(5)=0.7;
%mu_Q(5)=0.22;
%mu_Q(4)=0.38;
%mu_Q(3)=0.45;
%mu_Q(2)=0.6;
%mu_Q(1)=0.7;


pai=mu_P./(mu_P+mu_Q);% Steady state st prob of state 1 (mean)

% Each arm is played once
for ii=1:n_arms
    if rand>mu_P(ii)
        mu_est(ii)=0;
        current_state(ii)=0;
    else
        mu_est(ii)=1;
        current_state(ii)=1;
    end
        played(ii)=played(ii)+1;
end


for ii = (n_arms+1):n_trials
    for x=1:n_arms
      UCB_est(x)=mu_est(x)+sqrt(343*log(ii)/played(x));% Calculate UCB for each arm
    end
    [UCB_max,arm_selected]=max(UCB_est);% Index of max UCB arm
    
    if current_state(arm_selected)==0 
        % Update currrent state and reward of each arm
        if rand>mu_P(arm_selected)
            reward=0;
            current_state(arm_selected)=0;
        else
            reward=1;
            current_state(arm_selected)=1;
        end
    else
        if rand>mu_Q(arm_selected)
            reward=1;
            current_state(arm_selected)=1;
        else
            reward=0;
            current_state(arm_selected)=0;
        end
    end
    % Calculate regret
    regret(iteration,ii)=regret(iteration,ii-1)+max(pai)-pai(arm_selected);
    regret_1og(iteration,ii)=regret(iteration,ii)/log(ii);
    % Update sample mean of selected arm
    mu_est(arm_selected)=(played(arm_selected)*mu_est(arm_selected)+reward)/(played(arm_selected)+1);
    played(arm_selected)=played(arm_selected)+1;
end
 for arm=1:n_arms
 played_iteration(iteration,arm)=played(arm);
 end
end
 % Mean and Std deviation
 regret_avg=sum(regret,1)/iteration_total;
 regret_sd=std(regret);
 
 %Confidence interval 95%
 ci_up=regret_avg+1.96*regret_sd/10;
 ci_low=regret_avg-1.96*regret_sd/10;
 ci_width=ci_up-ci_low;
figure
plot(trial,regret_avg)
hold
%plot(trial,ci_up)
%plot(trial,ci_low)
%plot(trial,ci_width);