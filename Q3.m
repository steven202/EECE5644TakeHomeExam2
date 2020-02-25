
% for k=1:10
%     counts(:,:,k)=run();
% end
%%

count=run(10);
dummy_array=[1,2,3,4,5,6];
disp(count);
%%

count1=run(50);
dummy_array=[1,2,3,4,5,6];
disp(count1);

%%

count2=run(50);
dummy_array=[1,2,3,4,5,6];
disp(count2);
%%
figure;
plot(dummy_array,count(1,:),'--gs');hold on;
plot(dummy_array,count(2,:),'--bs');hold on;
plot(dummy_array,count(3,:),'--rs');

w1=find(count(1,:)==max(count(1,:)));
w2=find(count(2,:)==max(count(2,:)));
w3=find(count(3,:)==max(count(3,:)));
fprintf("the components win most frequently in D_{10} is %d, in D_{100} is %d, in D_{1000} is %d\n", w1,w2,w3);
plot(w1,max(count(1,:)),'kx','MarkerSize',10);hold on;
plot(w2,max(count(2,:)),'kx','MarkerSize',10);hold on;
plot(w3,max(count(3,:)),'kx','MarkerSize',10);hold on;

legend('D_{10}','D_{100}','D_{1000}');
title('which Gaussian components get the most counts');
xlabel('number of Gaussian components'); ylabel('counts per 100 executions');
hold on;
%%
% for k=1:10
%     counts(:,:,k)=run(30);
% end
% %%
% figure;
% for w=1:10
%     plot(dummy_array,counts(1,:,k),'--gs');hold on;
%     plot(dummy_array,counts(2,:,k),'--bs');hold on;
%     plot(dummy_array,counts(3,:,k),'--rs');hold on;
% end
% legend('D_{10}','D_{100}','D_{1000}')
%%
function count=run(d)
    s = RandStream('mlfg6331_64'); 
    count=zeros(3,6);
    n1=360;n2=500;n3=1400;
    for m=1:100
        alpha_true = [ 0.2, 0.25, 0.25, 0.3];
        mu_true = [-10 0 10 1;0 2 0 -3];
        Sigma_true(:,:,1) = [3 1;1 20];
        Sigma_true(:,:,2) = [7 1;1 2];
        Sigma_true(:,:,3) = [4 1;1 16];
        Sigma_true(:,:,4) = [5 1;1 2];
        for b = 1:d
            x1 = randGMM(10,alpha_true,mu_true,Sigma_true);
            x2 = randGMM(100,alpha_true,mu_true,Sigma_true);
            x3 = randGMM(1000,alpha_true,mu_true,Sigma_true);

            x1_train(1,:) = datasample(s,x1(1,:),n1,'Replace',true);
            x1_train(2,:) = datasample(s,x1(2,:),n1,'Replace',true);
            x2_train(1,:) = datasample(s,x2(1,:),n2,'Replace',true);
            x2_train(2,:) = datasample(s,x2(2,:),n2,'Replace',true);
            x3_train(1,:) = datasample(s,x3(1,:),n3,'Replace',true);
            x3_train(2,:) = datasample(s,x3(2,:),n3,'Replace',true);

            x1_validate(1,:) = datasample(s,x1(1,:),n1,'Replace',true);
            x1_validate(2,:) = datasample(s,x1(2,:),n1,'Replace',true);
            x2_validate(1,:) = datasample(s,x2(1,:),n2,'Replace',true);
            x2_validate(2,:) = datasample(s,x2(2,:),n2,'Replace',true);
            x3_validate(1,:) = datasample(s,x3(1,:),n3,'Replace',true);
            x3_validate(2,:) = datasample(s,x3(2,:),n3,'Replace',true);
            for iii =1:6
                [x1_alpha,x1_mu,x1_sigma]=init_params(m,x1_train);
                [x2_alpha,x2_mu,x2_sigma]=init_params(m,x2_train);
                [x3_alpha,x3_mu,x3_sigma]=init_params(m,x3_train);
                [logLikelihood1,alpha1,mu1,Sigma1]=EMforGMM(x1_train,x1_alpha,x1_mu,x1_sigma,x1_validate);
                [logLikelihood2,alpha2,mu2,Sigma2]=EMforGMM(x2_train,x2_alpha,x2_mu,x2_sigma,x2_validate);
                [logLikelihood3,alpha3,mu3,Sigma3]=EMforGMM(x3_train,x3_alpha,x3_mu,x3_sigma,x3_validate);
                likelihoods1(b,iii)=logLikelihood1;
                likelihoods2(b,iii)=logLikelihood2;
                likelihoods3(b,iii)=logLikelihood3;
            end
        end
        ave1=mean(likelihoods1,1);
        ave2=mean(likelihoods2,1);
        ave3=mean(likelihoods3,1);
        winner1=find(ave1==max(ave1),1);
        winner2=find(ave2==max(ave2),1);
        winner3=find(ave3==max(ave3),1);
        
        count(1,winner1)=count(1,winner1)+1;
        count(2,winner2)=count(2,winner2)+1;
        count(3,winner3)=count(3,winner3)+1;
    end
end
%%
function [alpha,mu,Sigma]=init_params(M,x)
    delta = 1e-2; % tolerance for EM stopping criterion
    regWeight = 1e-10; % regularization parameter for covariance estimates
    N=size(x,2);
    % Initialize the GMM to randomly selected samples
    alpha = ones(1,M)/M;
    shuffledIndices = randperm(N);
    mu = x(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
    [~,assignedCentroidLabels] = min(pdist2(mu',x'),[],1); % assign each sample to the nearest mean
    for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
        Sigma(:,:,m) = cov(x(:,find(assignedCentroidLabels==m))') + regWeight*eye(2,2);
    end
end
%%
function [logLikelihood,alpha,mu,Sigma]=EMforGMM(x,alpha,mu,Sigma,x_validate)
% Generates N samples from a specified GMM,
% then uses EM algorithm to estimate the parameters
% of a GMM that has the same nu,mber of components
% as the true GMM that generates the samples.

close all,
delta = 1e-2; % tolerance for EM stopping criterion
regWeight = 1e-10; % regularization parameter for covariance estimates
%%%%%%%%%%%%%%%%%%%%%% generate data %%%%%%%%%%%%%%%%%
% % Generate samples from a 3-component GMM
% mu_true = [-10 0 10 1;0 2 0 -3];
% 
% 
% [d,M] = size(mu_true); % determine dimensionality of samples and number of GMM components

t = 0; %displayProgress(t,x,alpha,mu,Sigma);

Converged = 1; % Not converged at the beginning
while ~Converged && t<=1000
    for l = 1:M
        temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(x,mu(:,l),Sigma(:,:,l));
    end
    plgivenx = temp./sum(temp,1);
    alphaNew = mean(plgivenx,2);
    w = plgivenx./repmat(sum(plgivenx,2),1,N);
    muNew = x*w';
    for l = 1:M
        v = x-repmat(muNew(:,l),1,N);
        u = repmat(w(l,:),d,1).*v;
        SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
    end
    Dalpha = sum(abs(alphaNew-alpha'));
    Dmu = sum(sum(abs(muNew-mu)));
    DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
    Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
    alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
    t = t+1; 
end
logLikelihood = sum(log(evalGMM(x_validate,alpha,mu,Sigma)));
end

%%
function x = randGMM(N,alpha,mu,Sigma)
    d = size(mu,1); % dimensionality of samples
    cum_alpha = [0,cumsum(alpha)];
    u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
    for m = 1:length(alpha)
        ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1)); 
        x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
    end
end

%%
function x = randGaussian(N,mu,Sigma)
    % Generates N samples from a Gaussian pdf with mean mu covariance Sigma
    n = length(mu);
    z =  randn(n,N);
    A = Sigma^(1/2);
    x = A*z + repmat(mu,1,N);
end

%%
function [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2)
    x1Grid = linspace(floor(rangex1(1)),ceil(rangex1(2)),101);
    x2Grid = linspace(floor(rangex2(1)),ceil(rangex2(2)),91);
    [h,v] = meshgrid(x1Grid,x2Grid);
    GMM = evalGMM([h(:)';v(:)'],alpha, mu, Sigma);
    zGMM = reshape(GMM,91,101);
    %figure(1), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
end


%%
function gmm = evalGMM(x,alpha,mu,Sigma)
    gmm = zeros(1,size(x,2));
    for m = 1:length(alpha) % evaluate the GMM on the grid
        gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
    end
end

%%
function g = evalGaussian(x,mu,Sigma)
    % Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
    [n,N] = size(x);
    Sigma(isnan(Sigma))=-1000;
    invSigma = pinv(Sigma);
    C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
    E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
    g = C*exp(E);
end
