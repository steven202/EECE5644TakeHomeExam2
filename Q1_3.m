%% ================== Generate and Plot Training Set ================== %%
clear all; close all; clc;
rng('default');
rng(7);
n = 2;      % number of feature dimensions
N = 1000;   % number of iid samples

% parallel distributions
mu(:,2) = [-2;0]; Sigma(:,:,1) = [1 -0.9; -0.9 2]; 
mu(:,1) = [2;0]; Sigma(:,:,2) = [2 0.9;0.9 1];
%mu(:,1) = [3;0]; Sigma(:,:,1) = [5 0.1;0.1 .5]; 
%mu(:,2) = [0;0]; Sigma(:,:,2) = [.5 0.1;0.1 5];

% Class priors for class 0 and 1 respectively
p = [0.9,0.1]; 

% Generating true class labels
label = (rand(1,N) >= p(1))';
Nc = [length(find(label==0)),length(find(label==1))];

% Draw samples from each class pdf
x = zeros(N,n); 

for L = 0:1
    x(label==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc(L+1));
end

%Plot samples with true class labels
figure(1);
plot(x(label==0,1),x(label==0,2),'o',x(label==1,1),x(label==1,2),'+');
legend('Class 0','Class 1'); title('Training Data and True Class Labels');
xlabel('x_1'); ylabel('x_2'); hold on;

%% ======================== Logistic Regression ======================= %%
% Initialize fitting parameters
x = [ones(N, 1) x];

initial_theta = zeros(n+1+3, 1);
label=double(label);

% Compute gradient descent to get theta values
[theta, cost] = gradient_descent(x,N,label,initial_theta,1,4000);
%[theta2, cost2] = fminsearch(@(t)(cost_func(t, x, label, N)), initial_theta);
theta2=theta;
cost2=cost;
%f=@(x,y)x*theta2;
f=@(x,y)theta2(1)+theta2(2).*x+theta2(3).*y+theta2(4).*x.^2+theta2(5).*x.*y+theta2(6).*y.^2;
fimplicit(f, [min(x(:,2))-10, max(x(:,2))+10 , min(x(:,3))-10, max(x(:,3))+10]);
% % Choose points to draw boundary line
% min_max = [min(x(:,2))-2,  max(x(:,2))+2];                      
% % plot_x2(1,:) = (-1./theta(3)).*(theta(2).*min_max + theta(1));  
% y_min_max = (-1./theta2(3)).*(theta2(2).*min_max + theta2(1)); % fminsearch
% 
% % Plot decision boundary
% plot(min_max, y_min_max);  
 axis([min(x(:,2))-10, max(x(:,2))+10 , min(x(:,3))-10, max(x(:,3))+10]);
 legend('Class 0', 'Class 1', 'Classifier (fminsearch)');

% Plot cost function
% figure(2); plot(cost);
% title('Calculated Cost');
% xlabel('Iteration number'); ylabel('Cost');

%% ====================== Generate Test Data Set ====================== %%
N_test = 10000;

% Generating true class labels
label_test = (rand(1,N_test) >= p(1))';
Nc_test = [length(find(label_test==0)),length(find(label_test==1))];

% Draw samples from each class pdf
x_test = zeros(N_test,n); 
for L = 0:1
    x_test(label_test==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc_test(L+1));
end

%% ========================= Test Classifier ========================== %%
% Coefficients for decision boundary line equation
%coeff(1,:) = polyfit([min_max(1), min_max(2)], [y_min_max(1), y_min_max(2)], 2); %fminsearch
%coeff(2,:) = polyfit([min_max(1), min_max(2)], [plot_x2(2,1), plot_x2(2,2)], 1); 
% Decide based on which side of the line each point is on
y_test = generate_y(x_test,theta2,10000);
decision = (y_test >= 0.5);

ind00 = find(decision==0 & label_test==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label_test==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label_test==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label_test==1); p11 = length(ind11)/Nc(2); % probability of true positive
p_error = [p10,p01]*Nc_test'/10000;
fprintf('Total error (classifier using gradient dscent): %d\n',p_error);
figure(3);
plot(x_test(ind00,1),x_test(ind00,2),'o'); hold on,
plot(x_test(ind10,1),x_test(ind10,2),'oc'); hold on,
plot(x_test(ind01,1),x_test(ind01,2),'+m'); hold on,
plot(x_test(ind11,1),x_test(ind11,2),'+r'); hold on,

f=@(x,y)theta2(1)+theta2(2).*x+theta2(3).*y+theta2(4).*x.^2+theta2(5).*x.*y+theta2(6).*y.^2;
fimplicit(f, [min(x_test(:,1))-10, max(x_test(:,1))+10 , min(x_test(:,2))-10, max(x_test(:,2))+10]);
title('Test Data Classification (from gradient dscent)');
axis([min(x_test(:,1))-10, max(x_test(:,1))+10 , min(x_test(:,2))-10, max(x_test(:,2))+10]);
legend('Class 0 Correct Decisions','Class 0 Wrong Decisions','Class 1 Wrong Decisions','Class 1 Correct Decisions','Classifier'); 
 
%plot_test_data(decision(:,1), label_test, Nc_test, p, 3, x_test, [min(x(:,2))-10, max(x(:,2))+10] , [min(x(:,3))-10, max(x(:,3))+10]);


%% ============================ Functions ============================= %%
function [theta, cost] = gradient_descent(x, N, label, theta, alpha, num_iters)
    cost = zeros(num_iters, 1);
    x(:,4)=x(:,2).*x(:,2);
    x(:,5)=x(:,2).*x(:,3);
    x(:,6)=x(:,3).*x(:,3);
    for i = 1:num_iters % while norm(cost_gradient) > threshold
        h = 1 ./ (1 + exp(-x*theta));	% Sigmoid function   
        cost(i) = (-1/N)*((sum(label' * log(h)))+(sum((1-label)' * log(1-h))));
        cost_gradient = (1/N)*(x' * (h - label));
        theta = theta - (alpha.*cost_gradient); % Update theta
    end
end

function cost = cost_func(theta, x, label,N)
    x(:,4)=x(:,2).*x(:,2);
    x(:,5)=x(:,2).*x(:,3);
    x(:,6)=x(:,3).*x(:,3);
    h = 1 ./ (1 + exp(-(x*theta)));	% Sigmoid function
    cost = (-1/N)*((sum(label' * log(h)))+(sum((1-label)' * log(1-h))));
end
function y = generate_y(x, theta, N)
    x = [ones(N, 1) x];
    x(:,4)=x(:,2).*x(:,2);
    x(:,5)=x(:,2).*x(:,3);
    x(:,6)=x(:,3).*x(:,3);
    y = x*theta;	% Sigmoid function
end
function plot_test_data(decision, label, Nc, p, fig, x, min_max, plot_x2)
    ind00 = find(decision==0 & label==0); % true negative
    ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % false positive
    ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % false negative
    ind11 = find(decision==1 & label==1); % true positive

    % Plot decisions and decision boundary
    figure(fig);
    plot(x(ind00,1),x(ind00,2),'og'); hold on,
    plot(x(ind10,1),x(ind10,2),'or'); hold on,
    plot(x(ind01,1),x(ind01,2),'+r'); hold on,
    plot(x(ind11,1),x(ind11,2),'+g'); hold on,
    plot(min_max, plot_x2);
    axis([min_max(1), min_max(2), min(x(:,2))-2, max(x(:,2))+2])
    legend('Class 0 Correct Decisions','Class 0 Wrong Decisions','Class 1 Wrong Decisions','Class 1 Correct Decisions','Classifier');
end