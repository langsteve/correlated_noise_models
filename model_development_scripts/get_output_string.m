function output = get_output_string(number)

leading_character = '';
if number < 0
    leading_character = '-';
end

number = abs(number);

if number < 0.001
    number_power = ceil(log10(number) - 1);
    if number_power == -Inf
        output = '0';
    else
        raw_number = 10 ^ (log10(number) - number_power);
        if abs(number_power) < 10
            output = [leading_character, num2str(raw_number, '%.5f'), 'E-0', num2str(abs(number_power))];
        else
            output = [leading_character, num2str(raw_number, '%.5f'), 'E-', num2str(abs(number_power))];
        end
    end
elseif number >= 0.001 && number < 0.01
    output = [leading_character, num2str(number, '%.7f')];
elseif number >= 0.01 && number < 0.1
    output = [leading_character, num2str(number, '%.6f')];
elseif number >= 0.1 && number < 1000
    output = [leading_character, num2str(number, '%.5f')];
else
    number_power = ceil(log10(number) - 1);
    raw_number = 10 ^ (log10(number) - number_power);
    num_digits = min(15, 5 + number_power);
    if number_power < 10
        output = [leading_character, num2str(raw_number, ['%.', num2str(num_digits), 'f']), ...
                  'E+0', num2str(number_power)];
    else
        output = [leading_character, num2str(raw_number, ['%.', num2str(num_digits), 'f']), ...
                  'E+', num2str(number_power)];
    end
end

end
