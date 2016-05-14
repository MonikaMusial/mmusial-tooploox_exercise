function J = computeCost(X, y, theta)

m = length(y); 
J = 0;
prediction = 0;
prediction = X *theta;
sqrErrors= (prediction - y).^2;
J=1/(2*m)*sum(sqrErrors);


end
