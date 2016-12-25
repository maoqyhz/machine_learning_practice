data = load('data2.txt');
fprintf('Running Gradient Descent ...\n');

X = data(:,1:2);
y = data(:,3);
m = length(y);

[X,avg,sigma] = normalize(X);
X = [ones(m,1) , X];

theta = zeros(3,1);
alpha = 0.01;
max_iter = 10000;

J_history = zeros(max_iter,1);
for iter = 1:max_iter
	theta = theta - alpha / m * X' * (X * theta - y); 
    J_history(iter) = sum((X * theta - y).^2) / (2*m);    
    fprintf('iter:%d ------ Error:%f\n',iter,J_history(iter));
end

disp('Theta found by gradient descent:');
disp(theta);

figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Estimate the price of a 1650 sq-ft, 3 br house
x = [1650 3];
price = [1 (([1650 3] - avg) ./ sigma)] * theta ;
disp(price); 