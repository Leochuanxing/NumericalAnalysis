function [a, s] = myrand(low, high)
a = low + rand(3, 4) * (high - low);
s = sumElements(a);

function s = sumElements(a)
% We should avoid using global as much as possible
% for you make not know who moves the value of this
% variable.
global v;
v = a(:);
s = sum(v);

