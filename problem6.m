function [time,  grad_norm , iteration] = problem6(dimension_n, contraction_factor,Goldstein_c, ...
    initial_alpha, termination_condition)
tic
n = dimension_n;
% Generate Rosenbrock function and gradient
[f, df, d2f] = problem3(n);
% Implement Algorithm 5.4
% Evaluate f(x0)
x = zeros(1, n);
x_cell = num2cell(x);
y = feval(f, x_cell{:});
grad_f = feval(df, x_cell{:});
% set up the termination condition
grad_norm = sqrt(grad_f*grad_f'); 
% record the number of iterations
iteration =0;
while grad_norm  >= termination_condition
   %Define tolerance
   epik = min(0.5, sqrt(grad_norm)) * grad_norm;
   z = zeros(1, n);
   r = grad_f;
   d = -r;
   % Evaluate Bk
   Bk = feval(d2f, x_cell{:});

   while r*r' >= epik^2
       dBkd = d * Bk * d';
       if dBkd <= 0
           if z == zeros(1, n)
               p = -grad_f;
               break
           else
               p = z;
               break
           end
       end
        % Store the previous r
        pre_r = r;
        al = pre_r * pre_r'/dBkd;
        z = z + al * d;
        r = pre_r + al * d * Bk;
        beta = r*r'/(pre_r*pre_r');
        d = -r + beta * d;
        p = z;
   end
   % Do the Armijo search
   rho = contraction_factor;
   c = Goldstein_c;
   alpha = initial_alpha;
   x_cell = num2cell(x + alpha.* p);
   while feval(f, x_cell{:}) > y + c.*alpha.*grad_f * p'
        alpha = alpha*rho;
        x_cell = num2cell(x + alpha.* p);
%         alpha_iter = alpha_iter + 1;
   end
    % update x
    x = x + alpha.*p;
    % Update y and grad_f
    %x_cell = num2cell(x);
    y = feval(f, x_cell{:});
    grad_f = feval(df, x_cell{:});
    % update grad_norm_square
    grad_norm = sqrt(grad_f * grad_f');
    iteration = iteration + 1;
end
time  = toc;
end
