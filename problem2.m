function problem2
% construct a 50*50 positive definite matrix
% with evenly distributed eigen values
even_eigen_values = (0.2:0.2:10);
simple_matrix = zeros(50);
for i = 1:50
    simple_matrix(i, i) = even_eigen_values(i);
end
% Construt a simple 50*50 positive definite matrix
% The eigen values are grouped into two clusters
% one cluster is around 1. We use rand to generate
% those eigen values. The other cluster is around 10

cluster1 = 0.5 + rand(1, 25);
cluster2 = 9.5 + rand(1, 25);
cluster_all = [cluster1, cluster2];
simple_matrix_clustered = zeros(50);
for i= 1:50
    simple_matrix_clustered(i, i) = cluster_all(i);
end

[log_norm1, iteration1] = CG_standard(simple_matrix);
[log_norm2, iteration2] = CG_standard(simple_matrix_clustered);

plot(log_norm1, iteration1)
plot(log_norm2, iteration2)
% solve the problems with standard CG method
% Lets iterate 50 times
    function [log_norm, iteration] = CG_standard(A)
        % create empty containers to contain the results
        log_norm = zeros(1, 49);
        iteration = 1:49;
        b = ones(n, 1);
        inverse_matrix = zeros(50);
        for k = 1: 50
            inverse_matrix(k,k) = 1/(A(k,k));
        end
        exact_solution = inverse_matrix * b;

        x=zeros(n, 1);
        r = A*x - b;
        p = -r;
        for j = iteration
            r_temp = r;
            alpha = (r_temp' * r_temp)/(p'* A * p);
            x = x + alpha * p;
            r = r_temp + alpha * A * p;
            beta = (r' * r)/(r_temp' * r_temp);
            p = -r + beta * p;
            
            log_norm(j)= log(x' * A * exact_solution);
        end
    end
end