function g = sigmoid(z)
g = zeros(size(z));

SIGMOID = @(z) 1./(1 + exp(-z));
g=SIGMOID(z);

end
