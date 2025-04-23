clear
close all


f = @(z) sin(4*z).*(abs(z).^2+1);
N = 100;
S = spdiags(ones(2*N+3,1),1,2*N+3,2*N+3); S = S';

xpts=-2:0.01:2;    ypts=-0.4:0.025:0.4;
zpts=kron(xpts,ones(length(ypts),1))+1i*kron(ones(1,length(xpts)),ypts(:));    zpts=zpts(:);		% complex points where we compute pseudospectra

RES=0*zpts+1;

pf = parfor_progress(length(find(abs(abs(f(zpts))-1)<2)));
% pf = parfor_progress(length(zpts));
pfcleanup = onCleanup(@() delete(pf));

for jj=1:length(zpts)
    if abs(abs(f(zpts(jj)))-1)<2
        B = S-f(zpts(jj))*S';
        % B = B(1:2*N,1:2*N); % finite section
        B = B(:,2:end-1);
        RES(jj) = svds(B,1,'smallest');
        parfor_progress(pf);
    end
   
end

RES=reshape(RES,length(ypts),length(xpts));

%% Plot the results
v=(10.^(-20:0.2:0));
% v=(10.^(-50:0.4*8:0));
f=figure
contourf(reshape(real(zpts),length(ypts),length(xpts)),reshape(imag(zpts),length(ypts),length(xpts)),log10(max(real(RES),min(v))),log10(v),'LineColor',[1,1,1]*0.3,...
    'linewidth',1,'linestyle','-','ShowText','off');
cbh=colorbar;
cbh.Ticks=log10(10.^(-20:1:0));
% cbh.Ticks=log10(10.^(-30:5:0));
cbh.TickLabels=["1e-20","1e-19","1e-18","1e-17","1e-16","1e-15","1e-14","1e-13","1e-12","1e-11",...
    "1e-10","1e-9","1e-8","1e-7","1e-6","1e-5","1e-4","1e-3","1e-2","1e-1","1"];
% cbh.TickLabels=["1e-30","1e-25","1e-20","1e-15","1e-10","1e-5","1"];
clim([-2,0])
% clim([-30,0])
colormap bone
ax=gca; ax.FontSize=14;
hold on
xlabel('$\mathrm{Re}(z)$','interpreter','latex')
ylabel('$\mathrm{Im}(z)$','interpreter','latex')
ax=gca; ax.FontSize=18;
box on
grid minor
set(gca,'layer','top');
f.Position=[160.0000   97.6667  560.0000*2  420.0000]
