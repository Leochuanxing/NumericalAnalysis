function [time,  grad_norm , iteration] = problem5(dimension_n, contraction_factor,Goldstein_c, ...
    initial_alpha, termination_condition, algorithm )
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
% set the initial Hessien as an identity matrix
H = eye(n);
% creat an identity for later usage
I = eye(n);
% set up the termination condition
termination = termination_condition^2;
grad_norm_square = grad_f*grad_f'; 
% record the number of iterations
iteration =0;
while grad_norm_square  >= termination
    % search for alpha by the Armijo backtracking method
    % Store grad_f in pre_grad_f
    pre_grad_f = grad_f;
    p = pre_grad_f * (-H);
    % lets set the initial value of alpha as 1
    alpha = initial_alpha;
    x_cell = num2cell(x + alpha.* p);
%     alpha_iter = 0;
    while feval(f, x_cell{:}) > y + c.*alpha.*pre_grad_f * p'
        alpha = alpha*rho;
        x_cell = num2cell(x + alpha.* p);
%         alpha_iter = alpha_iter + 1;
    end
    % We add the following block to adjust the initial_alpha
    % This block can speed up the searching by a magnitude.
%     if alpha_iter >= 5
%         initial_alpha = 0.5*initial_alpha;
%     elseif alpha_iter <= 1
%         initial_alpha = 2* initial_alpha;
%     end
    % update x
    x = x + alpha.*p;
    % Update y and grad_f
    %x_cell = num2cell(x);
    y = feval(f, x_cell{:});
    grad_f = feval(df, x_cell{:});
    % update grad_norm_square
    grad_norm_square = grad_f * grad_f';
    % Calculate sk
    sk = alpha.*p;
    %Calculate yk
    yk = grad_f - pre_grad_f;
    % DFP, BFGS or SR1 to update H
    if algorithm == 'D'
        rh = yk * sk';
        % make sure rh != 0, if rh == 0, we reset H = I.
        if rh == 0
            H = I;
        else
            H = H - (H*(yk')*yk*H)/(yk*H*yk') + (sk' * sk)/rh;
        end
    elseif algorithm == 'B'
        rh = yk * sk';
        % make sure rh != 0, if rh == 0, we reset H = I.
        if rh == 0
            H = I;
        else
        rhok = 1/ rh;
        H = (I - rhok .* (sk' * yk))*H*(I - rhok.*(yk' * sk))+ rhok * (sk') * sk;
        end
    elseif algorithm == 'S'
        denominator = (sk - yk*H ) * yk';
        if denominator == 0
            H = I;
        else
            H = H + ((sk -yk * H)') * (sk - yk * H)./denominator; 
        end
    end
    % Update iteration
    iteration = iteration + 1;
end
    % report sqrt norm square
    grad_norm = sqrt(grad_norm_square);
    time = toc;
end
