%% Initialization
clear ; close all; clc

  %% =============1) Reading in the data and analysing the basic statistics   =================== 

  file_contents = importdata('data.csv',',',0);
x=size(file_contents.textdata);
y=size(file_contents.data);
id=file_contents.textdata;
v=file_contents.data;   
n=[1:size(file_contents.data,2)];
 
av = mean(v);
figure 
  plot(n,mean(v),'--gs',...
    'LineWidth',1,...
    'MarkerSize',2,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[0.5,0.5,0.5])
  ylabel('number of views recorded')
  xlabel('hours after publishing a video')
  title('avarage number of views')
  
 
  %% =============2) the distribution of v(168)   ===================  
figure
plot(v(:,168));
  ylabel('number of views recorded')
  xlabel('video id')
  title('number of views in 168 hour after publishing a video')
  %% =============3) the distribution of the log transformed v(168)   ===================
figure
plot(log(v(:,168)));
  ylabel('number of views recorded')
  xlabel('video id')
  title('log transformed number of views in 168 hour')
fprintf('This is a normal distribution\n');
  %% =============4) removing outliers   ===================
  A=v(:,168);
  B=log(v(:,168));
sigma=std(B); 
av168 = mean(B);
min = av168 - 3.*sigma;
max= av168 +3.*sigma;

[row,col] = find(B>max);
v(row,:)=[];
[row2,col] = find(B<min);
v(row2,:)=[];


  %% =============5) correlation coecients   ===================
Cor = zeros(24,1);
for i=1:24,
    fprintf('Correlation Coeffitient between log(v(%d)) and v(168)',i);
   corrcoef(log(v(:,i)),v(:,168))    
end;

 
  %% =============6) splitting the dataset into training and test sets   ===================
   q=size(v,1);
   m = randn(q,1);
   w = randsample(length(m),q*0.9);
   
  TrainSet = v(w,:);
  TestSet = v(~ismember(1:q,w),:);



  %% =============7)running linear regression model  ===================
fprintf('linear regression model  \n');
Cost=zeros(24,1);
ThetaOne = zeros(2,size(TrainSet,2));

for i=1:(size(TrainSet,2)-1),
    
    input = [ones(size(TrainSet,1), 1), TrainSet(:,i)];
    ThetaOne(:,i)=normalEqn(input, TrainSet(:,168));
    Cost(i)= computeCost(input, TrainSet(:,168), ThetaOne(:,i));
    
end;


  %% =============8) linear regression model with multiple inputs, ===================
  
fprintf('running linear regression model with multiple inputs\n');

Theta = zeros(size(TrainSet,2),size(TrainSet,2)-1);

for i=1:(size(TrainSet,2)-1),
    
    input = [ones(size(TrainSet,1), 1), TrainSet(:,1:i)];
    Theta(1:i+1,i)=normalEqn(input, TrainSet(:,168));
    
end;

  %% =============9) mean Relative Squared Error ===================
  
 mRSE=zeros(24,1);
 mRSEOne=zeros(24,1);
 for i=1:24,
     inputMult = [ones(size(TestSet,1), 1), TestSet(:,1:i)];
     predictions= inputMult * Theta(1:i+1,i);
     mRSE(i)= (1/size(TestSet,1))*sum((predictions./TestSet(:,168)-1).^2);
     
     inputOne = [ones(size(TestSet,1), 1), TestSet(:,i)];
     predictionsOne= inputOne * ThetaOne(:,i);
     mRSEOne(i)= (1/size(TestSet,1))*sum((predictionsOne./TestSet(:,168)-1).^2);
 end;
 
figure
hold on;
plot(1:24,mRSE);
  ylabel('mRSE')
  xlabel('reference time')
  title('Performance of linear regression models for n hours measured as mRSE')
 
plot(1:24,mRSEOne);

legend('Linear Regression','Multiple Input Linear Regression')
hold off;
  
  