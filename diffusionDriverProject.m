clear
close all

N = 4;
%p = 0.05;
%m = 4;
beta = 0.01;


node = makeLNProject(N);
%node = makeglobal(N);
%node = makeSW(N,m,0.1);


% A = adjacency(node);
% A = ham2adj(N);
% node = adj2node(A);

[N,e,avgdegree,maxdegree,mindegree,numclus,meanclus,Lmax,L2,LmaxL2] = clusterstats(node);

disp(' ')
displine('Number of nodes = ',N)
disp(strcat('Number of edges = ',num2str(e)))
disp(strcat('Mean degree = ',num2str(avgdegree)))
displine('Maximum degree = ',maxdegree)
disp(strcat('Number of clusters = ',num2str(numclus)))
disp(strcat('mean cluster coefficient = ',num2str(meanclus)))
disp(' ')
disp(strcat('Lmax = ',num2str(Lmax)))
disp(strcat('L2 = ',num2str(L2)))
disp(strcat('Lmax/L2 = ',num2str(LmaxL2)))
disp(' ')

 [A,degree,Lap] = adjacency(node);
 
 [V,D] = eig(Lap);
 
 for loop = 1:N
     eigval(loop) = D(loop,loop);
 end
 
 figure(1)
 plot(eigval)
 title('Eigenvalues')
 
 % initial values 

 % mass of tanks
 mtank = 1;
 % spring constant
 springk = 1;

 % masses of gasses
 c = zeros(N,1);
 c(1) = 5;
 % initial frequencies
 %omegas = zeros(N,1);
 %omegas(1) = 3;
 natFreqs = zeros(N,1)

 % initial angles
 phis = zeros(N,1);

 
 % eigvector decomposition
 for eigloop = 1:N
     Vtemp = V(:,eigloop)
     v(eigloop) = sum(c.*Vtemp)
 end
 
 omegax = [];

 % time loop (using eigenvalues, not really iterative)
 Ntime = 100;
 for timeloop = 1:Ntime       % 200
     
     for nodeloop = 1:N
         
         temp = 0;
         for eigloop = 1:N 
             
             temp = temp + V(nodeloop,eigloop)*v(eigloop)*exp(-eigval(eigloop)*beta*(timeloop-1));
                  
         end    % end eigloop
         concentration(timeloop,nodeloop) = temp;

     end    % endnodeloop
     

 end % end timeloop
 
 %More ways to plot eigenvalue-time data
 figure(2)
 imagesc(real(log(concentration)))
 colormap(jet)
 colorbar
 caxis([-10 0])
 title('Log Concentrations vs. time AAA')
 xlabel('Node Number')
 
 figure(3)
 plot(concentration(100,:))
 title('Ending Concentrations BBB')
 xlabel('Node Number')
 
 x = 0:Ntime-1;
 h = colormap(jet);
 figure(4)
 for nodeloop = 1:N
     rn = round(rand*63 + 1);
     y =  concentration(:,nodeloop)+0.001;
     semilogy(x,y,'Color',h(rn,:))
      hold on
 end
 hold off
 title('Concentrations vs. time CCC')
 
 
 % Now try the discrete-time-map approach
 
 %initial concentrations
 c0 = c;
 %change in time
 dt = 1;
 %Floquet Multiplier
 M = eye(N,N) - beta*Lap*dt;
 
%  for timeloop = 1:200   %40     
%      c = (M^timeloop)*c0;
%      Con(timeloop,:) = c';
%  end
 
 cold = [5;3;4;2]%c0;
 cnew = [5;3;4;2]%c0;
 omega0 = 3;

 avgMass = sum(c)/N + mtank;
 avgFreq = omega0/sqrt(avgMass/mtank);
 tloops = 400;
 for timeloop = 1:tloops  %40
     
     % Modelling Diffusion with a small time step, using floquet multiplier
     % matrix M
     %cnew = (M^1)*cold;
     Con(timeloop,:) = cnew';
     

     % Frequency Work
     natFreqs = omega0./(sqrt((cold+mtank)./mtank))';%sqrt(springk./(mtank+cnew))';
     g = timeloop*3/tloops;
     dphis = zeros(N,1);
     for philoop = 1:N
        dphis = dphis + g/N * sin((phis(philoop)-phis).*A(:,philoop));
     end
     %dphis = (dphis + natFreqs') * dt - avgFreq;
     dphisdt = (dphis + natFreqs') - avgFreq;
     phis = phis + dphisdt * dt;
     dPhisdt(timeloop,:) = dphisdt;
     Phis(timeloop,:) = phis;

     cold = cnew;
 end

 x = 0:1:(tloops-1);   % 0:5:199
 h = colormap(jet);
 
 figure(5)
 for nodeloop = 1:N
     rn = round(rand*63 + 1);
     y =  Con(:,nodeloop)+0.001;
     %semilogy(x,y,'Color',h(rn,:),'LineWidth',1.1)
     semilogy(x,y,'k','LineWidth',1.2)
     title('Discrete time (Concentration)')
     hold on
 end
 hold off
 set(gcf,'Color','White')
 
 figure(6)
 for nodeloop = 1:N
     rn = round(rand*63 + 1);
     y =  dPhisdt(:,nodeloop)+0.001;
     %semilogy(x,y,'Color',h(rn,:),'LineWidth',1.1)
     plot(x,y,'k','LineWidth',1.2)
     title('Discrete time (Frequencies)')
     hold on
 end
 hold off
 set(gcf,'Color','White')
 
 figure(7)
 for nodeloop = 1:N
     rn = round(rand*63 + 1);
     y =  Phis(:,nodeloop)+0.001;
     plot(x,y,'k','LineWidth',1.2)
     title('Discrete time (Phases)')
     hold on
 end
 hold off
 set(gcf,'Color','White')
 
 
 
 
 
 
 
 