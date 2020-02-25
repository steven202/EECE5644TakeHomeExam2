%%%%%% The code is partially borrowed from the shared folder provided by the Professor Deniz.
%%%%%% Some of the code is from volunteers in the shared folder. 
%%%%%% Many thanks to professor and volunteers. 
% Expected risk minimization with 2 classes
clear; close all; %clc;

rng('default');
rng(1);

n = 2; % number of feature dimensions
N = 10000; % number of iid samples
p = [0.9,0.1];
% parallel distributions
mu(:,2) = [-2;0]; Sigma(:,:,1) = [1 -0.9; -0.9 2]; 
mu(:,1) = [2;0]; Sigma(:,:,2) = [2 0.9;0.9 1];
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % save up space
% Draw samples from each class pdf
for l = 0:1
    %x(:,label==l) = randGaussian(Nc(l+1),mu(:,l+1),Sigma(:,:,l+1));
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end

lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
ratio = evalGaussian(x,mu(:,2),Sigma(:,:,2))-evalGaussian(x,mu(:,1),Sigma(:,:,1));
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);
% socre < gamma - 0 
% score >=gamma - 1
decision = (discriminantScore >= log(gamma)); 

disp(gamma);

ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
%p(error) = [p10,p01]*Nc'/N; % probability of error, empirically estimated
p_error_point = [p10,p01]*Nc'/N;
%%%%%%%%%%%%% finish %%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% ROC Curve %%%%%%%%%%%%%%%%%%%%%
figure(4), clf;
dim0=1000;
dim1=10;
gammas = linspace(0, dim1,dim0);
gammas = [gammas, inf];
threshold=zeros(2,dim0+1);

for i=1:dim0+1
    decision_temp = (discriminantScore >= log(gammas(i))); 
    ind11_temp = find(decision_temp==1 & label==1); p11_temp = length(ind11_temp)/Nc(2); % probability of true positive
    ind10_temp = find(decision_temp==1 & label==0); p10_temp = length(ind10_temp)/Nc(1); % probability of false positive
    threshold(1,i)=p11_temp;
    threshold(2,i)=p10_temp;
end

Y = threshold(1,:);
X = threshold(2,:);
plot(X,Y,'-','LineWidth',2), hold on;
title('Roc curve'),
xlabel('probability false positive'), ylabel('probability of true positive');

hold on;
plot(p10, p11,'*r');
text(p10,p11,'(0.0039, 0.8814)','FontSize',13);
fprintf('threshold gamma: %d\n',gamma);
fprintf('(false positive: %d, true positive: %d)\n',p10,p11);
fprintf('minimum probability of error: %d\n', p_error_point);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,

% Draw the decision boundary
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantScoreGridValues = log(evalGaussian([h(:)';v(:)'],mu(:,2),Sigma(:,:,2)))-log(evalGaussian([h(:)';v(:)'],mu(:,1),Sigma(:,:,1))) - log(gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
figure(1), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
% including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Equilevel contours of the discriminant function' ), 
title('Data and their classifier decisions versus true labels'),
xlabel('x_1'), ylabel('x_2'), 