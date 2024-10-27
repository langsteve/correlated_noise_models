function make_plots(sampling_interval, filter_duration, num_sv, t, minimum_eig_val, ...
                    KF_covariance_using_AR2_models, KF_covariance_using_WGN_models, ...
                    true_covariance_using_AR2_models)

% Function header.
% This function generates figures 10-13 in [1].

%-- Inputs:
%   sampling_interval                : A positive scalar indicating the time interval between adjacent GPS
%                                      measurements, in seconds.
%
%   filter_duration                  : A positive scalar indicating the duration of the Kalman filter in
%                                      seconds.
%
%   num_sv                           : A positive scalar indicating the number of satellites used in the KF
%                                      solution (which is fixed at 7 for the baseline estimation example).
%
%   t                                : A (1xn) vector used for making plots of the covariance ellipses.
%
%   minimum_eig_val                  : An (nx1) vector containing the minimum eigenvalue of the covariance
%                                      error matrix Delta after each KF measurement update.
%
%   KF_covariance_using_AR2_models   : An (nx82) matrix where, for a given row, the first column is the
%                                      measurement time and columns 2 through 82 are the vectorized form of
%                                      the KF covariance matrix for the position and cycle ambiguity states
%                                      when using second-order AR models for correlated noise.
%
%   KF_covariance_using_WGN_models   : An (nx82) matrix where, for a given row, the first column is the
%                                      measurement time and columns 2 through 82 are the vectorized form of
%                                      the KF covariance matrix for the position and cycle ambiguity states
%                                      when using WGN models for correlated noise.
%
%   true_covariance_using_AR2_models : An (nx82) matrix where, for a given row, the first column is the
%                                      measurement time and columns 2 through 82 are the vectorized form of
%                                      the true covariance matrix for the position and cycle ambiguity states
%                                      when then KF uses second-order AR models for correlated noise.
%
%-- References:
%      [1] S. Langel, O. Garcia Crespillo, and M. Joerger, "Frequency-domain modeling of correlated Gaussian
%          noise in Kalman filtering," IEEE Trans. Aerosp. Electron. Syst., vol. xx, no. xx, pp. xx-xx,
%          November, 2024, doi: 10.1109/TAES.2024.3442775.


time_vector = (0 : sampling_interval : filter_duration)';

% Plot the minimum eigenvalue of the covariance error matrix, Delta (Fig. 10 in [1]).
figure
hold on; box on; grid on;
set(gcf, 'color', 'white');
set(gca, 'linewidth', 1.5, 'fontsize', 16, 'TickLabelInterpreter', 'latex');
plot(time_vector, minimum_eig_val, 'k', 'linewidth', 2);
xlim([0, filter_duration]);
ylim([0, 1]);
xlabel('$\textrm{Elapsed Time (sec)}$', 'fontsize', 16, 'Interpreter', 'latex');
ylabel('$\lambda_{\textrm{min}} \textrm{ of } \mbox{\boldmath $\Delta$}_{\xi} \textrm{  (cm}^{\mathrm{2}})$', ...
        'fontsize', 16, 'Interpreter', 'latex');
xlabel('$\textrm{Elapsed Time (sec)}$', 'fontsize', 16, 'Interpreter', 'latex');
xticks([0 100 200 300 400 500 600]);
yticks([0 0.25 0.5 0.75 1]);
xticklabels({'$0$', '100', '$200$', '$300$', '$400$', '$500$', '$600$'});
yticklabels({'$0$', '$0.25$', '$0.5$', '$0.75$', '$1$'});

% Plot the KF and true covariance matrices at elapsed times of 60 and 540 seconds for the x and z components
% of the relative position vector (Fig. 11 in [1]).
figure('units', 'normalized', 'outerposition', [0 0 1 1]);
hold on; box on; grid on;
set(gcf, 'color', 'white');
set(gca, 'linewidth', 1.5, 'fontsize', 16, 'TickLabelInterpreter', 'latex', 'Position', ...
    [0.12 0.46 0.29 0.48]);
xlim([-60, 60]);
ylim([-75, 75]);
xticks([-60 -40 -20 0 20 40 60]);
yticks([-75 -50 -25 0 25 50 75]);
xlabel('$x\textrm{-component of } \Delta\textbf{\emph{x}} \hspace{1mm} (\textrm{cm})$', 'fontsize', 16, ...
       'interpreter','latex');
ylabel('$z\textrm{-component of }\Delta\textbf{\emph{x}} \hspace{1mm} \textrm{(cm)}$', 'fontsize', 16, ...
       'Interpreter', 'latex');

time_index_plot = 60;
ind = abs(KF_covariance_using_AR2_models(:, 1) - time_index_plot) < sampling_interval / 1000;
P = reshape(true_covariance_using_AR2_models(ind, 2 : end)', 3 + num_sv - 1, 3 + num_sv - 1);
Sigma = reshape(KF_covariance_using_AR2_models(ind, 2 : end)', 3 + num_sv - 1, 3 + num_sv - 1);
P = P([1, 3], [1, 3]);
Sigma = Sigma([1, 3], [1, 3]);

[V, D] = eig(P);
lam_1 = D(1, 1); lam_2 = D(2, 2);
true_cov_ellipse = V * [sqrt(lam_1) * cos(t); sqrt(lam_2) * sin(t)];
plot(true_cov_ellipse(1, :), true_cov_ellipse(2, :), 'color', [0.65, 0.65, 0.65], 'Marker', 's', ...
     'MarkerSize', 8, 'MarkerIndices', 1 : 3 : length(t), 'linewidth', 2);

[V, D] = eig(Sigma);
lam_1 = D(1, 1); lam_2 = D(2, 2);
bounding_cov_ellipse = V * [sqrt(lam_1) * cos(t); sqrt(lam_2) * sin(t)];
plot(bounding_cov_ellipse(1, :), bounding_cov_ellipse(2, :), 'color', 'k',  'Marker', 's', ...
     'MarkerSize', 8, 'MarkerIndices', 1 : 3 : length(t), 'linewidth', 2);

time_index_plot = 540;
ind = abs(KF_covariance_using_AR2_models(:, 1) - time_index_plot) < sampling_interval / 1000;
P = reshape(true_covariance_using_AR2_models(ind, 2 : end)', 3 + num_sv - 1, 3 + num_sv - 1);
Sigma = reshape(KF_covariance_using_AR2_models(ind, 2 : end)', 3 + num_sv - 1, 3 + num_sv - 1);
P = P([1, 3], [1, 3]);
Sigma = Sigma([1, 3], [1, 3]);

[V, D] = eig(P);
lam_1 = D(1, 1); lam_2 = D(2, 2);
true_cov_ellipse = V * [sqrt(lam_1) * cos(t); sqrt(lam_2) * sin(t)];
plot(true_cov_ellipse(1, :), true_cov_ellipse(2, :), 'color', [0.65, 0.65, 0.65], 'Marker', '.', ...
     'MarkerSize', 14, 'MarkerIndices', 1 : 4 : length(t), 'linewidth', 2);

[V, D] = eig(Sigma);
lam_1 = D(1, 1); lam_2 = D(2, 2);
bounding_cov_ellipse = V * [sqrt(lam_1) * cos(t); sqrt(lam_2) * sin(t)];
plot(bounding_cov_ellipse(1, :), bounding_cov_ellipse(2, :), 'color', 'k', 'Marker', '.', ...
     'MarkerSize', 14, 'MarkerIndices', 1 : 4 : length(t), 'linewidth', 2);

legend('$\vphantom{b^{b^b}}\vphantom{b_{b_b}}\textbf{P}_{60}$', ...
       '$\vphantom{b^{b^b}}\vphantom{b_{b_b}}\bf{\Sigma}_{\rm{60}}$', ...
       '$\vphantom{b^{b^b}}\vphantom{b_{b_b}}\textbf{P}_{540}$', ...
       '$\vphantom{b^{b^b}}\vphantom{b_{b_b}}\bf{\Sigma}_{\rm{540}}$', ...
       'Location', 'southeast', 'fontsize', 16, 'NumColumns', 2, 'Interpreter', 'latex');

% Plot the KF and true covariance matrices at elapsed times of 60 and 540 seconds for the second and sixth
% components of the cycle ambiguity vector (Fig. 12 in [1]).
figure('units', 'normalized', 'outerposition', [0 0 1 1]);
hold on; box on; grid on;
set(gcf, 'color', 'white');
set(gca, 'linewidth', 1.5, 'fontsize', 16, 'Position', ...
    [0.112664257385101, 0.454712362301102, 0.282420808777470, 0.470287637698897], ...
    'TickLabelInterpreter', 'latex');
xlim([-75, 75]);
ylim([-75, 75]);
xticks([-75 -50 -25 0 25 50 75]);
yticks([-75 -50 -25 0 25 50 75]);
xlabel('$b_2 \textrm{ (cm)}$', 'fontsize', 16, 'interpreter','latex');
ylabel('$b_6 \textrm{ (cm)}$', 'fontsize', 16, 'Interpreter', 'latex');

time_index_plot = 60;
ind = abs(KF_covariance_using_AR2_models(:, 1) - time_index_plot) < sampling_interval / 1000;
P = reshape(true_covariance_using_AR2_models(ind, 2 : end)', 3 + num_sv - 1, 3 + num_sv - 1);
Sigma = reshape(KF_covariance_using_AR2_models(ind, 2 : end)', 3 + num_sv - 1, 3 + num_sv - 1);
P = P([5, 9], [5, 9]);
Sigma = Sigma([5, 9], [5, 9]);

[V, D] = eig(P);
lam_1 = D(1, 1); lam_2 = D(2, 2);
true_cov_ellipse = V * [sqrt(lam_1) * cos(t); sqrt(lam_2) * sin(t)];
plot(true_cov_ellipse(1, :), true_cov_ellipse(2, :), ...
     'color', [0.65, 0.65, 0.65], 'Marker', 's', 'MarkerSize', 8, ...
     'MarkerIndices', 1 : 3 : length(t), 'linewidth', 2);

[V, D] = eig(Sigma);
lam_1 = D(1, 1); lam_2 = D(2, 2);
bounding_cov_ellipse = V * [sqrt(lam_1) * cos(t); sqrt(lam_2) * sin(t)];
plot(bounding_cov_ellipse(1, :), bounding_cov_ellipse(2, :), ...
     'color', 'k',  'Marker', 's', 'MarkerSize', 8, ...
     'MarkerIndices', 1 : 3 : length(t), 'linewidth', 2);

time_index_plot = 540;
ind = abs(KF_covariance_using_AR2_models(:, 1) - time_index_plot) < sampling_interval / 1000;
P = reshape(true_covariance_using_AR2_models(ind, 2 : end)', 3 + num_sv - 1, 3 + num_sv - 1);
Sigma = reshape(KF_covariance_using_AR2_models(ind, 2 : end)', 3 + num_sv - 1, 3 + num_sv - 1);
P = P([5, 9], [5, 9]);
Sigma = Sigma([5, 9], [5, 9]);

[V, D] = eig(P);
lam_1 = D(1, 1); lam_2 = D(2, 2);
true_cov_ellipse = V * [sqrt(lam_1) * cos(t); sqrt(lam_2) * sin(t)];
plot(true_cov_ellipse(1, :), true_cov_ellipse(2, :), 'color', [0.65, 0.65, 0.65], 'Marker', '.', ...
     'MarkerSize', 14, 'MarkerIndices', 1 : 4 : length(t), 'linewidth', 2);

[V, D] = eig(Sigma);
lam_1 = D(1, 1); lam_2 = D(2, 2);
bounding_cov_ellipse = V * [sqrt(lam_1) * cos(t); sqrt(lam_2) * sin(t)];
plot(bounding_cov_ellipse(1, :), bounding_cov_ellipse(2, :), 'color', 'k', 'Marker', '.', ...
     'MarkerSize', 14, 'MarkerIndices', 1 : 4 : length(t), 'linewidth', 2);

legend('$\vphantom{b^{b^b}}\vphantom{b_{b_b}}\textbf{P}_{60}$', ...
       '$\vphantom{b^{b^b}}\vphantom{b_{b_b}}\bf{\Sigma}_{\rm{60}}$', ...
       '$\vphantom{b^{b^b}}\vphantom{b_{b_b}}\textbf{P}_{540}$', ...
       '$\vphantom{b^{b^b}}\vphantom{b_{b_b}}\bf{\Sigma}_{\rm{540}}$', ...
       'Location', 'southeast', 'fontsize', 16, 'NumColumns', 2, 'Interpreter', 'latex');

% Plot the percent increase in the predicted position error variance when using WGN models for correlated
% noise versus second-order AR models (Fig. 13 in [1]).
time_vector = KF_covariance_using_AR2_models(:, 1);
AR_data = sqrt(KF_covariance_using_AR2_models(:, [2, 12, 22]));
WGN_data = sqrt(KF_covariance_using_WGN_models(:, [2, 12, 22]));
percent_increase_x = (WGN_data(:, 1) - AR_data(:, 1)) ./ AR_data(:, 1) * 100;
percent_increase_y = (WGN_data(:, 2) - AR_data(:, 2)) ./ AR_data(:, 2) * 100;
percent_increase_z = (WGN_data(:, 3) - AR_data(:, 3)) ./ AR_data(:, 3) * 100;

figure('units', 'normalized', 'outerposition', [0 0 1 1]);
hold on; box on; grid on;
set(gcf, 'color', 'white');
set(gca, 'linewidth', 1.5, 'fontsize', 16, 'TickLabelInterpreter', 'latex', 'Position', ...
    [0.12, 0.46, 0.29, 0.48]);
xlim([0, 600]);
ylim([0, 75]);
xticks([0 100 200 300 400 500 600]);
yticks([0 15 30 45 60 75]);
xlabel('$\textrm{Elapsed Time (sec)}$', 'fontsize', 16, 'interpreter','latex');
ylabel(['$\displaystyle\frac{\sigma_{\fontsize{6}{0}\selectfont\textrm{WGN}} - '...
        '\sigma_{\fontsize{6}{0}\selectfont\textrm{AR}}}', ...
        '{\sigma_{\fontsize{6}{0}\selectfont\textrm{AR}}} \times 100$'], 'Interpreter', 'latex');

plot(time_vector, percent_increase_x, 'color', [0.65, 0.65, 0.65], 'linewidth', 2);
plot(time_vector, percent_increase_y, 'color', 'k', 'linewidth', 2);
plot(time_vector, percent_increase_z, 'LineStyle', '--', 'color', [0.65, 0.65, 0.65], 'linewidth', 2);

legend('$\vphantom{b^b}\vphantom{b_b} x$', ...
       '$\vphantom{b^b}\vphantom{b_b} y$', ...
       '$\vphantom{b^b}\vphantom{b_b} z$', ...
       'Location', 'northeast', 'fontsize', 16, 'NumColumns', 1, 'Interpreter', 'latex');

end
