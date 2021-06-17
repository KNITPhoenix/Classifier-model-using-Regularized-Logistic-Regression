function [J, grad] = costFunctionReg(theta, X, y, lambda)
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

a=0;
for i=1:m,
	a=a+(-(y(i)*log(sigmoid(theta' * X(i,:)')))-((1-y(i))*log(1-sigmoid(theta' * X(i,:)'))));
end;
J=(1/m)*a;
b=0;
for i=2:length(theta),
	b=b+(theta(i)^2);
end;
J=J+((lambda/(2*m))*b)

grad(1)=(1/m)*(X(:,1)'*((sigmoid(theta' * X'))' - y));
grad(2:size(X,2))=((1/m)*(X(:,2:size(X,2))'*((sigmoid(theta' * X'))' - y))) + ((lambda/m)*theta(2:size(X,2)));

end