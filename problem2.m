function problem2
% construct a 50*50 positive definite matrix
% with evenly distributed eigen values
even_eigen_values = (0.2:0.2:10);
simple_matrix = zeros(50);
for i = 1:50
    simple_matrix(i, i) = even_eigen_values(i);
end
% Construt a simple 50*50 positive definite matrix
% The eigen values are grouped into three clusters.
% They are clustered around 1, 10 and 20.

cluster1 = 0.5 + rand(1, 20)*0.5;
cluster2 = 9.5 + rand(1, 20)*0.5;
cluster3 = 19.5 + rand(1, 10)* 0.5;
cluster_all = [cluster1, cluster2, cluster3];
simple_matrix_clustered = zeros(50);
for i= 1:50
    simple_matrix_clustered(i, i) = cluster_all(i);
end

[log_norm1, iteration1] = CG_standard(simple_matrix);
[log_norm2, iteration2] = CG_standard(simple_matrix_clustered);

% plot the figures
figure
plot(iteration1, log_norm1, iteration2, log_norm2)
title('Performance of CG for different distributions of eigenvalues')
xlabel('Number of iterations')
ylabel('log(||x-x*||_A^2)')
legend({'Evenly distributed eigenvalues', 'Clustered eigenvalues'},...
        'Location', 'southwest')

% Solve the problems with standard CG method
% The number of iteration is 6.
    function [log_norm, iteration] = CG_standard(A)
        % Create empty containers to contain the results
        b = ones(50, 1);

        exact_solution = A\b;

        log_norm = zeros(1, 6);
        iteration = 1:6;
        x=zeros(50, 1);
        r = A*x - b;
        p = -r;
        for j = iteration
            r_temp = r;
            alpha = (r_temp' * r_temp)/(p'* A * p);
            x = x + alpha * p;
            r = r_temp + alpha * A * p;
            beta = (r' * r)/(r_temp' * r_temp);
            p = -r + beta * p;
            
            log_norm(j)= log((x-exact_solution)' * A * (x-exact_solution));
        end
    end
end