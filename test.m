function x = test
x = cell(1, 2);
for i = 1:2
    x{1, i} = sprintf('x%d', i);
end
x = sym(x, 'real');
end