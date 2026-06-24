% classdef_demo.m — OOP classdef demo
classdef Point
    properties
        x = 0
        y = 0
    end
    methods
        function obj = Point(x, y)
            obj.x = x;
            obj.y = y;
        end
        function d = distance(obj)
            d = sqrt(obj.x^2 + obj.y^2);
        end
    end
end

p = Point(3, 4);
disp(p.x);
disp(p.y);
disp(p.distance());
