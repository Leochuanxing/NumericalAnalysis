function [time,  grad_norm , iteration] = problem7(dimension_n, contraction_factor,Goldstein_c, ...
    initial_alpha,termination_condition)
tic
delta = 1;
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
   normr = sqrt(r * r');
   d = -r;
   % Evaluate Bk
   Bk = feval(d2f, x_cell{:});
   % r may be small enough, reduce the epik to make sure the following loop
   % can go on
   while normr < epik
       p = z;
   end
   % set up some parameters
   % We can tune the algorithm by change the following parameters.
   deltaMax = 1;
   ita = 0.2;
   while normr >= epik
       % Store the previous r
       pre_r = r;
       % Store the z to pre_z
       pre_z = z;
       dBkd = d * Bk * d';
       if dBkd <= 0
           [p, delta] = TrustRegion_p_delta(d,f,y,grad_f,Bk, pre_z, dBkd, deltaMax, delta, ita);
          break
       end
        al = pre_r * pre_r'/dBkd;
        z = pre_z + al * d;
        if z * z' > delta^2
            [p, delta] = TrustRegion_p_delta(d,f,y,grad_f,Bk, pre_z, dBkd, deltaMax, delta, ita);
            break
        end
        r = pre_r + al * d * Bk;
        normr = r * r';
        if normr < epik
            p = z;
            break
        end
        beta = r*r'/(pre_r*pre_r');
        d = -r + beta * d;
   end
   
%    if p ~= zeros(1, n)
%        % Do the Armijo search
%        rho = contraction_factor;
%        c = Goldstein_c;
%        alpha = initial_alpha;
%        x_cell = num2cell(x + alpha.* p);
%        while feval(f, x_cell{:}) > y + c.*alpha.*grad_f * p'
%             alpha = alpha*rho;
%             x_cell = num2cell(x + alpha.* p);
%     %         alpha_iter = alpha_iter + 1;
%        end
        % update x
        x = x + p;
        x_cell = num2cell(x);
%         x = x + alpha.*p;
        % Update y and grad_f
        %x_cell = num2cell(x);
        y = feval(f, x_cell{:});
        grad_f = feval(df, x_cell{:});
        % update grad_norm_square
        grad_norm = sqrt(grad_f * grad_f');
%    end
    %Update the iteration 
    iteration = iteration + 1;
end
    % The function TrustRegion_p_delta should report a proper searching
    % direction p and the up dated delta
    function [p, delta] = TrustRegion_p_delta(d,f, y ,grad_f, B, prez, dBkd, deltaMax, delta, ita)
        % find the vactor in the form of q = prez + td, which minimize mk(q)
        b = 2 * grad_f * d' + d * B * prez' + prez * B * d';
        normz = sqrt(prez*prez');
        normd = sqrt(d * d');
        if normz ==0
            % In this case we use the step in the direction of d with step
            % length delta
            q = (delta/normd) * d;    
        else 
            % In the followin, we discusse the update step q to minimize m(q) in different
            % casses. Fortunately, all those can be expressed in the closed
            % form. 
            cos  = (prez * d')/(normz) * normd;
            t_max = -normz * cos + sqrt(normz^2 * (cos^2-1) + delta ^2);
            if dBkd < 0
                t_lim = -b / (2*dBkd);
                if abs(t_lim) > abs(t_max - t_lim)
                    t = 0;
                else
                    t= t_max;
                end
                q = prez + t * d;
            elseif dBkd == 0
                if b >= 0
                    q = prez;
                else
                    q = prez + t_max * d;
                end
            elseif dBkd >0
                t_lim = -b / (2*dBkd);
                if abs(t_lim) < abs(t_max - t_lim)
                    t = 0;
                else
                    t= t_max;
                end
                q = prez + t * d;
            end
        end
        
       % Update delta and choose a proper report value
       % the following algorithm is similar to algorithm 4.1 with a small
       % alternation that if the improvement is negative, we discard the
       % update, shrink the radius of the trust region and optimize again.
        xq = x + q;
        xq_cell = num2cell(xq);
        rhok = y- feval(f, xq_cell{:});
        if rhok <= 0
            p = zeros(1, n);
            delta = 0.25 * delta;
            return
        else
            denom = -(grad_f * q' + 0.5 * q * B * q');
            rhok = rhok/denom; 
            %Update delta, the radius of the trust region
            if rhok < 0.25
                delta = 0.25 * delta;
            elseif rhok > 0.75
                delta = min(2*delta, deltaMax);
            end        
            % Report proper value of p
            if rhok < ita
                p = zeros(1, n);
            else
                p = q;
            end 
        end
    end
time  = toc;
end