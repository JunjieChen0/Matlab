% function_test.m — Function definition and call
function result = add(a, b)
    result = a + b;
end

function [q, r] = mydiv(a, b)
    q = floor(a / b);
    r = a - q * b;
end

x = add(3, 4);
disp(x);

[q, r] = mydiv(17, 5);
disp(q);
disp(r);
