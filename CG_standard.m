function [Iteration, residual] = CG_standard(n)
A = Hilbert_matrix(n);
b = ones(n, 1);
x=zeros(n, 1);
r = A*x - b;
p = -r;
Iteration = 0;
while sqrt(r'*r) >= 1e-6
    r_temp = r;
    alpha = (r_temp' * r_temp)/(p'* A * p);
    x = x + alpha * p;
    r = r_temp + alpha * A * p;
    beta = (r' * r)/(r_temp' * r_temp);
    p = -r + beta * p;
    Iteration = Iteration + 1;   
end
residual = sqrt(r'*r);

    

function A = Hilbert_matrix(n)
A = zeros(n);
for i = 1:n
    for j = 1:n
        A(i, j) = 1/(i + j - 1);
    end
end


