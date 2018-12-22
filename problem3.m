function [f, grad_f,hessen_f]=problem3(n)
% generate the symbolic Rosenbrock function, gradient and Hessian matrix
% creat the variables.
x = cell(1, n);
for i=1:n
    x{1, i} = sprintf('x%d', i);
end
% x = x(:);
x = sym(x, 'real');
% Create Rosenbrock function
Rosenbrock = sym(0);
for j = 1:(n-1)
    Rosenbrock = Rosenbrock + 100 * (x(j+1) - x(j)^2)^2 + (1-x(j))^2;
end
grad_Rosenbrock = jacobian(Rosenbrock, x);
hessen_Rosenbrock = jacobian(grad_Rosenbrock, x);
f = matlabFunction(Rosenbrock);
grad_f = matlabFunction(grad_Rosenbrock);
hessen_f = matlabFunction(hessen_Rosenbrock);
end