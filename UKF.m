clc
clear
close all
%% Definizione della scala temporale:
Tc=0.5;
N=500;
t=(0:N-1)*Tc;
Toss=t(end);

%% Definizione del modello:
rho_0 = 2; % lb*sec^2/(ft^4)
g = 32.2; %ft/sec^2
k0= 2*1e4; %ft
var_v = sqrt(1e4); %ft^2
var_wi = sqrt(0);
M = 1e5; %ft
a = M;

par = [rho_0, g, k0, var_v, var_wi, M, a]' ;

%% Simulazione del modello:

% dinamica del sistema:
x0_sys = [3*1e5 -2*1e4 0.001]';

f1 = inline('x_1k + x_2k + wk','x_1k','x_2k','wk');
f2 = inline('rho_0*exp(-x_1k/k)*x_3k*(x_2k).^2 - g + wk','rho_0','k','g','x_1k', 'x_2k' ,'x_3k','wk');
f3 = inline('xv3 + wk','xv3','wk');

% trasformazione d'uscita:
h = inline('sqrt(M^2 + (x_1k - a).^2) + vk', 'M','x_1k','a','vk');
%% Condizioni iniziali del filtro:
x0_hat = x0_sys;

P0 = diag([1e6 4*1e6 10]);
%% Scelta dei punti sigma:

x_k_post = x0_sys;

P_k_post = P0;

nx = size(x_k_post,1);
ny = 1;

x_hat_k1 = zeros(nx,2*nx);

x1_hat_k = zeros(1,2*nx);
x2_hat_k = zeros(1,2*nx);
x3_hat_k = zeros(1,2*nx);

x_hat_k = zeros(nx,2*nx);

P_ak_priori = zeros(nx,nx, 2*nx);
P_axy = zeros(nx,ny, 2*nx);
P_ay = zeros(ny,ny);
y_hat_k = zeros(1,2*nx);

x_tilde = zeros(nx, 2*nx);
x_tilde_priori = zeros(nx, 2*nx);

x_k_priori = zeros(nx,1);

fact(3, 3) = 0;
fact_priori(3, 3) = 0;

y_hat = 0;

x_hist = [];
y_hist=[];

for k = 1:N
    
    x_k = x_k_post; %lo stato corrente è lo stato precedente nella nuova iterazione:
    x_k1 = x_k;   
    P_k1_post = P_k_post;
    
    fact = chol(nx*P_k1_post)';
    
    noise_proc = normrnd(0, var_wi, 3, 1);
    Q_k = cov(noise_proc);
    
    noise_meas = normrnd(0, var_v, 1, 1);
    R_k =  cov(noise_meas);
        
    for i = 1:nx
        
        for j = 1:nx
            x_tilde(i,j) = fact(i,j);
            x_tilde(i, j + nx) = - fact(i,j);
        end
        
        x_hat_k1(:,i) = x_k1 + x_tilde(:,i);
        x_hat_k1(:,i + nx) = x_k1 + x_tilde(:,i + nx);
        
        x1_hat_k(i) = feval(f1, x_hat_k1(1,i), x_hat_k1(2,i), noise_proc(1));
        x1_hat_k(i + nx) = feval(f1, x_hat_k1(1,i + nx), x_hat_k1(2,i + nx), noise_proc(1));
        
        x2_hat_k(i) = feval(f2, rho_0, k0, g, x_hat_k1(1,i), x_hat_k1(2,i), x_hat_k1(3,i), noise_proc(2));
        x2_hat_k(i + nx) = feval(f2, rho_0, k0, g, x_hat_k1(1,i + nx), x_hat_k1(2,i + nx), x_hat_k1(3,i + nx), noise_proc(2));
        
        x3_hat_k(i) = feval(f3, x_hat_k1(3,i),noise_proc(3));
        x3_hat_k(i + nx) = feval(f3, x_hat_k1(3,i + nx),noise_proc(3));
        
        x_hat_k = [ x1_hat_k; ...
            x2_hat_k; ...
            x3_hat_k];
                
        
    x_k_priori = (sum(x_hat_k, 2)/(2*nx)); %sum(A,dim)
    
    P_ak_priori(:,:, i) =  (x_hat_k(:,i) - x_k_priori)*( x_hat_k(:,i) - x_k_priori)';

    P_ak_priori(:,:, i + nx) =  (x_hat_k(:,i + nx) - x_k_priori)*( x_hat_k(:,i + nx) - x_k_priori)';
    
    P_k_priori = sum(P_ak_priori,3)/(2*nx) + Q_k;
        
    end
           
    fact_priori = chol(nx*P_k_priori)';
    
     for i = 1:nx
        for j = 1:nx
            x_tilde_priori(i,j) = fact_priori(i,j);
            x_tilde_priori(i, j + nx) = - fact_priori(i,j);
        end
        
          x_hat_k1(:,i) = x_k1 + x_tilde_priori(:,i);
          x_hat_k1(:,i + nx) = x_k1 + x_tilde_priori(:,i + nx);
     
          y_hat_k(i) = feval(h, M , x_hat_k1(1,i), a, noise_meas);
          y_hat_k(i + nx) = feval(h, M , x_hat_k1(1,i + nx), a, noise_meas);

     
     y_hat = sum(y_hat_k)/(2*nx);
     
     P_ay(:,i) = ( y_hat_k(i) - y_hat )*( y_hat_k(i) - y_hat )';
     P_ay(:,i + nx) = ( y_hat_k(i + nx) - y_hat )*( y_hat_k(i + nx) - y_hat )';
     
     P_y = R_k + sum(P_ay,2)/(2*nx);
     
     P_axy(:, :, i) = (x_hat_k(:,i) - x_k_priori) * (y_hat_k(i) - y_hat)';
     P_axy(:, :, i + nx) = (x_hat_k(:,i + nx) - x_k_priori) * (y_hat_k(i + nx) - y_hat)';
     
     P_xy = sum(P_axy,3)/(2*nx);
     
     end
     K_k = P_xy*inv(P_y);
     
     x1_k = feval(f1, x_k(1), x_k(2), noise_proc(1));
     x2_k = feval(f2, rho_0, k0, g, x_k(1), x_k(2), x_k(3), noise_proc(2));
     x3_k = feval(f3, x_k(3), noise_proc(3));

     y = feval(h, M , x1_k, a, noise_meas);
     
     x_k_post = x_k_priori + K_k*( y - y_hat );
     
     P_k_post = P_k_priori - K_k*P_y*K_k';

     x_hist = [x_hist x_k_post];
     y_hist = [y_hist y];
     
end

y = feval(h, M, x_hist(1,:),a,noise_meas);
%%
figure(1)

plot(t,y_hist,'LineWidth',2);
hold on
plot(t,y,'r--','LineWidth',2)
legend('unfiltered', 'filtered')
