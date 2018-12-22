function [time, y, iteration] = problem4(dimension_n, contraction_factor,Goldstein_c, ...
    initial_alpha, termination_condition)
% lets set a timer
tic
n = dimension_n;
rho = contraction_factor;
c = Goldstein_c;
% Generate Rosenbrock function and gradient
[f, df, d2f] = problem3(n);
% Implement Algorithm 5.4
% Evaluate f(x0)
x = zeros(1, n);
x_cell = num2cell(x);
y = feval(f, x_cell{:});
grad_f = feval(df, x_cell{:});
p = -grad_f;
iteration =0;
termination = termination_condition^2;
grad_norm_square = grad_f*grad_f'; 
while grad_norm_square  >= termination
    % search for alpha by the Armijo backtracking method
    % Store grad_f in pre_grad_f
    pre_grad_f = grad_f;
    % lets set the initial value of alpha as 1
    alpha = initial_alpha;
    x_cell = num2cell(x + alpha.* p);
    alpha_iter = 0;
    while feval(f, x_cell{:}) > y + c.*alpha.*pre_grad_f * p'
        alpha = alpha*rho;
        x_cell = num2cell(x + alpha.* p);
        alpha_iter = alpha_iter + 1;
    end
    % We add the following block to adjust the initial_alpha
    % This block can speed up the searching by a magnitude.
    if alpha_iter >= 5
        initial_alpha = 0.5*initial_alpha;
    elseif alpha_iter <= 1
        initial_alpha = 2* initial_alpha;
    end
    % update x
    x = x + alpha.*p;
    % Update y and grad_f
    %x_cell = num2cell(x);
    y = feval(f, x_cell{:});
    grad_f = feval(df, x_cell{:});
    % Update the searching direction
    beta = (grad_f * grad_f')/(pre_grad_f * pre_grad_f');
    p = -grad_f + beta.* p;%
    %update the grad_norm_square
    grad_norm_square = grad_f*grad_f';
    % update the iteration number
    iteration = iteration + 1;   
end
time = toc;
% We report the time, the iteration number and the value of f.
end