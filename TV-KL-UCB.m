n_trials=100000;
n_arms=5;% number of arms
iteration_total=10;

%Total number of times an arm is played in an iteration
played_iteration=zeros(iteration_total,n_arms);
trial=zeros(1,n_trials);
regret=zeros(iteration_total,n_trials);% Regret for each trial

for ii=1:n_trials
    trial(ii)=ii;
end

for iteration=1:iteration_total
    count_KL=1;
    mu_P=zeros(1,n_arms);% Transition prob from state 0 to 1
    mu_Q=zeros(1,n_arms);% Transition prob from state 1 to 0
    pai=zeros(1,n_arms);% Steady state st prob of state 1 (mean)
    current_state=zeros(1,n_arms);% Running counter for current state
    mu_P_est=ones(1,n_arms);% Estimated mean for each arm
    mu_Q_est=ones(1,n_arms);% Estimated mean for each arm
    mu_est=zeros(1,n_arms);
    %mu_est=zeros(1,n_arms);
    played=zeros(2,n_arms);% number of times each arm/state picked
    KLUCB_est=zeros(1,n_arms);% Estimation of UCB for eac arm

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


    % Each arm-action pair is played once
    for ii=1:n_arms
        if rand>mu_P(ii)
            mu_P_est(ii)=0;
            mu_est(ii)=0;
            current_state(ii)=0;
            played(1,ii)=played(1,ii)+1;
        else
            mu_P_est(ii)=1;
            mu_est(ii)=1;
            current_state(ii)=1;
            played(1,ii)=played(1,ii)+1;
        end
        
       % disp(mu_P_est(ii));
        
      %  mu_est(ii)=mu_P_est(ii);
    end

%for ii=1:n_arms
%    if rand>mu_Q(ii)
 %       mu_Q_est(ii)=0;
 %       current_state(ii)=1;
 %       played(2,ii)=played(2,ii)+1;
 %   else
%       mu_Q_est(ii)=1;
%        current_state(ii)=0;
%        played(2,ii)=played(2,ii)+1;
%    end
%    disp(mu_Q_est(ii));    
 %  %    % mu_est(ii)=mu_P_est(ii)/(mu_P_est(ii)+mu_Q_est(ii));
%end


    for ii = (n_arms+1):(n_trials)
        for x=1:n_arms
            if mu_P_est(x)==1
                 mu_P_est(x)=0.999999;
            end
            if mu_P_est(x)==0
                mu_P_est(x)=0.000001;
            end
            if mu_Q_est(x)==1
                 mu_Q_est(x)=0.99999;
            end
            if mu_Q_est(x)==0
                mu_Q_est(x)=0.00001;
            end
          %  KL_iid=mu_P_est(x)*log(mu_P_est(x)/(1-mu_Q_est(x)))+(1-mu_P_est(x))*log((1-mu_P_est(x))/mu_Q_est(x));
          KL_iid=1-(mu_P_est(x)*(1-mu_Q_est(x)))^0.5-(mu_Q_est(x)*(1-mu_P_est(x)))^0.5;
          KL_iid=KL_iid^0.5;% use hellinger distance instead of KL distance
          KL_iid=KL_iid^0.2; %Squared Hellinger distance
          KL_iid=abs(mu_P_est(x)-1+mu_Q_est(x));% Total variation distance
          if (KL_iid >1/(ii^0.25))
         %  if (KL_iid >1/ii)
              count_KL=count_KL+1;
            if (current_state(x)==0) 
                %Newton-raphson method
                if mu_P_est(x)==1
                    mu_P_est(x)=0.999999;
                end
                if mu_P_est(x)==0
                    mu_P_est(x)=0.000001;
                end
               mu_P_est_var=mu_P_est(x)+0.00001;
               for it=1:50
                    mul=log(ii*(log(ii))^2+1)/(played(1,x)+played(2,x))-(mu_P_est(x)*log(mu_P_est(x)/mu_P_est_var)+(1-mu_P_est(x))*log((1-mu_P_est(x))/(1-mu_P_est_var)));
                    mul=mul*mu_P_est_var*(1-mu_P_est_var)/(mu_P_est(x)-mu_P_est_var);
                    mu_P_est_var= max(mu_P_est(x)+0.00001,mu_P_est_var-mul); 
                    mu_P_est_var=min(0.9999999, mu_P_est_var); 
                end
                %disp(mu_P_est_var);
                KLUCB_est(x)=mu_P_est_var/(mu_P_est_var+mu_Q_est(x));
            else 
              %  KL_iid_old=KL_iid;
                if mu_Q_est(x)==1
                    mu_Q_est(x)=0.999999;
                end
                if mu_Q_est(x)==0
                    mu_Q_est(x)=0.000001;
                end
                mu_Q_est_var=mu_Q_est(x)-0.00001;
                for it=1:50
                    mul=log(ii*(log(ii))^2+1)/(played(1,x)+played(2,x))-(mu_Q_est(x)*log(mu_Q_est(x)/mu_Q_est_var)+(1-mu_Q_est(x))*log((1-mu_Q_est(x))/(1-mu_Q_est_var)));
                    mul=mul*mu_Q_est_var*(1-mu_Q_est_var)/(mu_Q_est(x)-mu_Q_est_var);
                    mu_Q_est_var= min(mu_Q_est(x)-0.00001,mu_Q_est_var-mul); 
                    mu_Q_est_var=max(0.0001, mu_Q_est_var);    
                end
            %disp(mu_Q_est_var);
                KLUCB_est(x)=mu_P_est(x)/(mu_P_est(x)+mu_Q_est_var);
            end
     % KLUCB_est(x)=mu_est(x)+sqrt(100*log(ii)/played(x));% Calculate UCB for each arm
          else 
              %count_KL=count_KL+1;
              %fprintf('iteration %i\n', ii);
            %fprintf('arm %i\n', arm_selected);
              mu_est_var=mu_est(x)+0.00001;
                %Newton-raphson method
                if mu_est(x)==1
                    mu_est(x)=0.999999;
                end
                if mu_est(x)==0
                    mu_est(x)=0.000001;
                end
                for it=1:50
                    mul=log(ii*(log(ii))^2+1)/(played(1,x)+played(2,x))-(mu_est(x)*log(mu_est(x)/mu_est_var)+(1-mu_est(x))*log((1-mu_est(x))/(1-mu_est_var)));
                    mul=mul*mu_est_var*(1-mu_est_var)/(mu_est(x)-mu_est_var);
                    mu_est_var= max(mu_est(x)+0.00001,mu_est_var-mul); 
                    mu_est_var=min(0.9999999, mu_est_var); 
                end
                %disp(mu_P_est_var);
                KLUCB_est(x)=mu_est_var;
          end
        end
    
    [UCB_max,arm_selected]=max(KLUCB_est);% Index of max UCB arm
    
    if current_state(arm_selected)==0 
        % Update currrent state and reward of each arm
        if rand>mu_P(arm_selected)
            reward=0;
            current_state(arm_selected)=0;
        else
            reward=1;
            current_state(arm_selected)=1;
        end 
       mu_P_est(arm_selected)=(played(1,arm_selected)*mu_P_est(arm_selected)+reward)/(played(1,arm_selected)+1);
       mu_est(arm_selected)=((played(1,arm_selected)+played(2,arm_selected))*mu_est(arm_selected)+reward)/(played(1,arm_selected)+played(2,arm_selected)+1);
      played(1,arm_selected)=played(1,arm_selected)+1;
    else
        if rand>mu_Q(arm_selected)
            reward=0;
            reward_1=1;
            current_state(arm_selected)=1;
        else
            reward=1;
            reward_1=0;
            current_state(arm_selected)=0;
        end
        mu_Q_est(arm_selected)=(played(2,arm_selected)*mu_Q_est(arm_selected)+reward)/(played(2,arm_selected)+1);
        mu_est(arm_selected)=((played(1,arm_selected)+played(2,arm_selected))*mu_est(arm_selected)+reward_1)/(played(1,arm_selected)+played(2,arm_selected)+1);
        played(2,arm_selected)=played(2,arm_selected)+1;
    end
    %if (arm_selected==1)
        %KL_iid=mu_P_est(arm_selected)*log(mu_P_est(arm_selected)/(1-mu_Q_est(arm_selected)))+(1-mu_P_est(arm_selected))*log((1-mu_P_est(arm_selected))/mu_Q_est(arm_selected));
        %if (KL_iid< 1/ii)
       % disp(ii,'iteration');
        %    fprintf('iteration %i\n', ii);
         %   fprintf('arm %i\n', arm_selected);
        %    disp('yes');
        %disp(arm_selected,'arm');
       % else
        %    count_KL=count_KL+1;
        %    fprintf('iteration %i\n', ii);
        %    fprintf('arm %i\n', arm_selected);
        %end
    %end
    % Calculate regret
    regret(iteration,ii)=regret(iteration,ii-1)+max(pai)-pai(arm_selected);
    % Update sample mean of selected arm  
end

for arm=1:n_arms
  played_iteration(iteration,arm)=played(1,arm)+played(2,arm);
end

end

%MEan and Std dev
regret_avg=sum(regret,1)/iteration_total;
regret_sd=std(regret);
 
 %Confidence interval 95%
 ci_up=regret_avg+1.96*regret_sd/10;
 ci_low=regret_avg-1.96*regret_sd/10;
 ci_width=ci_up-ci_low;

 figure

plot(trial,regret_avg)
%hold
%plot(trial,ci_up)
%plot(trial,ci_low)

%semilogx(trial,regret)
title('KLUCB Strategy')
%xlim([1 n_trials])
xlabel('Trial')
ylabel('Regret')