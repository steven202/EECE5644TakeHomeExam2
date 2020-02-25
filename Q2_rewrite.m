% Aravind H. M. ("Arvin")       email: hmaravind1@gmail.com

clear; close all; clc;
rng('default')
gamma = 1;
I=eye(4);
%Input and initializations
N = 10;                       % Number of samples
Sigma = 1;          % Covariance - sample
SigmaV = mvnrnd(0,Sigma);                 % V8ariance - 0-mean Gaussian noise
nRealizations = 100;            % Number of realizations for the ensemble analysis
wTrue = mvnrnd(zeros(1,4),gamma^2*I);

% Input and initializations
mu = [0;0];                     % Mean - sample
 
gammaArray = logspace(-3,3,21);%logspace(-3,3,21);%10.^[-10:0.1:5];   % Array of gamma values
% gammaArray = 10.^[ceil(log10(eps)):-ceil(log10(eps))];


% MAP parameter estimation for an ensemble set of samples
tic;
[msqError, avMsqError, avPercentError, avAbsPercentError] = deal(zeros(nRealizations,length(gammaArray)));
for n = 1:nRealizations
    N = 10;%randi([20,30]);

    % Draw N samples of x from a Gaussian distribution
    x = (rand(N,1).*2.-1)';

    % Calculate y: quadratic in x + additive 0-mean Gaussian noise
    noise=mvnrnd(0,Sigma,N);
    x0=x';
    z(1,:)=x0.^3;
    z(2,:)=x0.^2;
    z(3,:)=x0;
    z(4,:)=ones(N,1);
    y = z'*wTrue'+noise;
    yTruth{1,n} =y;
    zQ = [ones(1,size(x,2)); x(1,:); x(1,:).^2; x(1,:).^3];

    % Compute z*z^T for linear and quadratic models
    for i = 1:N; zzTQ(:,:,i) = zQ(:,i)*zQ(:,i)'; end
    
    % MAP parameter estimation
    for i = 1:length(gammaArray)
        gamma = gammaArray(i);
        thetaMAP{1,n}(:,i) = (sum(zzTQ,3)+SigmaV^2/gamma^2*eye(size(zQ,1)))^-1*(zQ*y);
        yMAP{1,n}(:,i) = yFunc(x,thetaMAP{1,n}(:,i)',Sigma,N);
    end
    avMsqError(n,1:length(gammaArray))=sum((thetaMAP{1,n}-repmat(wTrue',1,length(gammaArray))).^2,1);
    % Mean squared error in y
    msqError(n,:) = N\sum((yMAP{1,n}-repmat(yTruth{1,n},1,length(gammaArray))).^2,1);
    
    % Average mean squared error of estimated parameters
%     avMsqError(n,1:length(gammaArray)) = length(wTrue')\sum((thetaMAP{1,n} - ...
%         repmat(wTrue',1,length(gammaArray))).^2);%./repmat(params,1,length(gammaArray))*100,1);
    
%     % Mean (over all parameters) of percent-error of estimated parameters
%     avPercentError(n,1:length(gammaArray)) = length(params)\sum((thetaMAP{1,n} - ...
%         repmat(params,1,length(gammaArray)))./repmat(params,1,length(gammaArray))*100,1);
%     
%     % Mean (over all parameters) of abs(percent-error) of estimated parameters
%     avAbsPercentError(n,1:length(gammaArray)) = length(params)\sum(abs((thetaMAP{1,n} - ...
%         repmat(params,1,length(gammaArray)))./repmat(params,1,length(gammaArray))*100),1);
end
toc;

%% Plot results - MAP Ensemble: mean squared error
fig = figure; fig.Position([1,2]) = [50,100];
fig.Position([3 4]) = 1.5*fig.Position([3,4]);
percentileArray = [1,25,50,75,100];
legend_array=["minimum"; "25%"; "median"; "75%"; "maximum"];


ax = gca; hold on; box on;
prctlMsqError = prctile(avMsqError,percentileArray,1);

med = median(avMsqError,1);
minimum = min(avMsqError,[],1);
maximum = max(avMsqError,[],1);
prctlMsqError(3,:)=med;
prctlMsqError(1,:)=minimum;
prctlMsqError(5,:)=maximum;

p=semilogx(ax,gammaArray,prctlMsqError,'LineWidth',2);
xlabel('gamma'); ylabel('norm 2 squared error between w_{true} and w_{map}'); ax.XScale = 'log';
lgnd = legend(ax,p,[legend_array,...
    repmat(' percentile',length(percentileArray),1)]); lgnd.Location = 'southwest';
pause;

% [~,ind] = min(abs(prctlMsqError(3,:)));
% plot(ax,gammaArray(ind),prctlMsqError(3,ind),'ro');
% lgnd = legend(ax,p,[num2str(percentileArray'),...
%     repmat(' percentile',length(percentileArray),1)]); lgnd.Location = 'southwest';
% pause;
%% Function to calculate y (without noise), given x and parameters
function y = yFunc(x,wTrue,Sigma,N)
    noise=mvnrnd(0,Sigma,N);
    x0=x;
    z(1,:)=x0.^3;
    z(2,:)=x0.^2;
    z(3,:)=x0;
    z(4,:)=ones(N,1);
    y = z'*wTrue'+noise;
end
