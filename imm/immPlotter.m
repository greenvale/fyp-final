clear
clc
close all

plot_sim = 1;
plot_species2 = 0;
pause_time = .01;
disp_particles = 1;
nspecies = 1;
nx = 30;
lx = 4.0*pi;
dx = lx/nx;
numElectrons = 10000;
numIons = 10000;
nt = 100;
skip = 1;

dt = 0.1*1.467;

x_centre = linspace(dx/2,lx-dx/2,nx);
x_face = linspace(0,lx,nx+1);

%% DATA IMPORT

if (disp_particles == 1)
    electron_pos_file = fopen('output/ho_electron_pos.txt','r');
    electron_pos_data = fscanf(electron_pos_file, '%f');
    fclose(electron_pos_file);

    electron_vel_file = fopen('output/ho_electron_vel.txt','r');
    electron_vel_data = fscanf(electron_vel_file, '%f');
    fclose(electron_vel_file);

    electron_pos_data = reshape(electron_pos_data, numElectrons,[]);
    electron_pos_data = electron_pos_data';

    electron_vel_data = reshape(electron_vel_data, numElectrons,[]);
    electron_vel_data = electron_vel_data';
end

lo_elec_file = fopen('output/lo_elec.txt','r');
lo_elec_data = fscanf(lo_elec_file, '%f');
fclose(lo_elec_file);

lo_elec_data = reshape(lo_elec_data, nx+1,[]);
lo_elec_data = lo_elec_data';

electron_lo_dens_file = fopen('output/lo_electron_dens.txt','r');
electron_lo_dens_data = fscanf(electron_lo_dens_file, '%f');
fclose(electron_lo_dens_file);

electron_lo_dens_data = reshape(electron_lo_dens_data, nx,[]);
electron_lo_dens_data = electron_lo_dens_data';

electron_lo_avgmom_file = fopen('output/lo_electron_avgmom.txt','r');
electron_lo_avgmom_data = fscanf(electron_lo_avgmom_file, '%f');
fclose(electron_lo_avgmom_file);

electron_lo_avgmom_data = reshape(electron_lo_avgmom_data, nx+1,[]);
electron_lo_avgmom_data = electron_lo_avgmom_data';

electron_lo_soln_file = fopen('output/lo_electron_soln.txt','r');
electron_lo_soln_data = fscanf(electron_lo_soln_file, '%f');
fclose(electron_lo_soln_file);

electron_lo_soln_data = reshape(electron_lo_soln_data, 2*nx,[]);
electron_lo_soln_data = electron_lo_soln_data';

electron_ho_cont_file = fopen('output/ho_electron_continuity.txt','r');
    electron_ho_cont_data = fscanf(electron_ho_cont_file, '%f');
    fclose(electron_ho_cont_file);

   electron_ho_cont_data = reshape(electron_ho_cont_data, nx,[]);
    electron_ho_cont_data = electron_ho_cont_data';

electron_ho_dens_file = fopen('output/ho_electron_dens.txt','r');
electron_ho_dens_data = fscanf(electron_ho_dens_file, '%f');
fclose(electron_ho_dens_file);

electron_ho_dens_data = reshape(electron_ho_dens_data, nx,[]);
electron_ho_dens_data = electron_ho_dens_data';

electron_ho_avgmom_file = fopen('output/ho_electron_avgmom.txt','r');
electron_ho_avgmom_data = fscanf(electron_ho_avgmom_file, '%f');
fclose(electron_ho_avgmom_file);

electron_ho_avgmom_data = reshape(electron_ho_avgmom_data, nx+1,[]);
electron_ho_avgmom_data = electron_ho_avgmom_data';

electron_ho_mom_file = fopen('output/ho_electron_mom.txt','r');
electron_ho_mom_data = fscanf(electron_ho_mom_file, '%f');
fclose(electron_ho_mom_file);

electron_ho_mom_data = reshape(electron_ho_mom_data, nx+1,[]);
electron_ho_mom_data = electron_ho_mom_data';

if nspecies >= 2
    if (disp_particles == 1)
        ion_pos_file = fopen('output/ho_ion_pos.txt','r');
        ion_pos_data = fscanf(ion_pos_file, '%f');
        fclose(ion_pos_file);

        ion_vel_file = fopen('output/ho_ion_vel.txt','r');
        ion_vel_data = fscanf(ion_vel_file, '%f');
        fclose(ion_vel_file);

        ion_pos_data = reshape(ion_pos_data, numIons,[]);
        ion_pos_data = ion_pos_data';

        ion_vel_data = reshape(ion_vel_data, numIons,[]);
        ion_vel_data = ion_vel_data';
    end

    lo_elec_file = fopen('output/lo_elec.txt','r');
    lo_elec_data = fscanf(lo_elec_file, '%f');
    fclose(lo_elec_file);

    lo_elec_data = reshape(lo_elec_data, nx+1,[]);
    lo_elec_data = lo_elec_data';

    ion_lo_dens_file = fopen('output/lo_ion_dens.txt','r');
    ion_lo_dens_data = fscanf(ion_lo_dens_file, '%f');
    fclose(ion_lo_dens_file);

    ion_lo_dens_data = reshape(ion_lo_dens_data, nx,[]);
    ion_lo_dens_data = ion_lo_dens_data';

    ion_lo_avgmom_file = fopen('output/lo_ion_avgmom.txt','r');
    ion_lo_avgmom_data = fscanf(ion_lo_avgmom_file, '%f');
    fclose(ion_lo_avgmom_file);

    ion_lo_avgmom_data = reshape(ion_lo_avgmom_data, nx+1,[]);
    ion_lo_avgmom_data = ion_lo_avgmom_data';
    
    ion_ho_cont_file = fopen('output/ho_ion_continuity.txt','r');
    ion_ho_cont_data = fscanf(ion_ho_cont_file, '%f');
    fclose(ion_ho_cont_file);

    ion_ho_cont_data = reshape(ion_ho_cont_data, nx,[]);
    ion_ho_cont_data = ion_ho_cont_data';

    ion_ho_dens_file = fopen('output/ho_ion_dens.txt','r');
    ion_ho_dens_data = fscanf(ion_ho_dens_file, '%f');
    fclose(ion_ho_dens_file);

    ion_ho_dens_data = reshape(ion_ho_dens_data, nx,[]);
    ion_ho_dens_data = ion_ho_dens_data';

    ion_ho_avgmom_file = fopen('output/ho_ion_avgmom.txt','r');
    ion_ho_avgmom_data = fscanf(ion_ho_avgmom_file, '%f');
    fclose(ion_ho_avgmom_file);

    ion_ho_avgmom_data = reshape(ion_ho_avgmom_data, nx+1,[]);
    ion_ho_avgmom_data = ion_ho_avgmom_data';

    ion_ho_mom_file = fopen('output/ho_ion_mom.txt','r');
    ion_ho_mom_data = fscanf(ion_ho_mom_file, '%f');
    fclose(ion_ho_mom_file);

    ion_ho_mom_data = reshape(ion_ho_mom_data, nx+1,[]);
    ion_ho_mom_data = ion_ho_mom_data';
end


lo_elec_energy = (lo_elec_data.^2);

lo_elec_energy_sum = sum(lo_elec_energy, 2);
lo_elec_energy_rate = (lo_elec_energy_sum(2:end) - lo_elec_energy_sum(1:end-1)) / dt;


if (plot_sim == 0)
    
% figure;
% [X_face,T] = meshgrid(x_face, dt*(1:nt)); 
% surf(X_face, T, elec_energy, 'EdgeColor', 'none');
% 
% figure;
% plot(dt*(1:nt-1), elec_energy_rate);
% 
% figure;
% plot(dt*(1:nt), elec_energy_sum);

figure;
hold on;
semilog_t = dt*(1:nt);
%semilogy(semilog_t(elec_energy_rate > 0), 1.3 * elec_energy_rate(elec_energy_rate > 0) / max(elec_energy_rate));
semilogy(dt*(1:nt), lo_elec_energy_sum);
x = dt*(1:nt-1);
y = max(lo_elec_energy_sum) * exp(x * (-0.155));
semilogy(x,y);
%ylim([1e-1 1]);
title('log plot of E field energy');
set(gca, 'YScale', 'log');

ylabel("E field energy");
xlabel("Time");

end

%% FIGURE 1

if (plot_sim == 1)
figure(1);
set(gcf,'posit',[8,14,1800,400]);
for i=1:skip:(nt+1)
    clf;
    sgtitle(['Timestep k=' num2str(i-1) ')']);
    subplot(1,5,1);
    hold on;
    title('Position vs velocity');
    if disp_particles == 1
    plot(electron_pos_data(i,:), electron_vel_data(i,:), 'b.');
    if (nspecies >= 2)
        plot(ion_pos_data(i,:), ion_vel_data(i,:), 'r.');
    end
    hold off;
    end
    ylabel("Velocity, v");
    xlabel("Position, x");
    
    subplot(1,5,2);
    %ylim([-2e-10, 2e-10]);
    title(['E field (t=' num2str(i-1) ')']);
    hold on;
    plot(x_face, lo_elec_data(i,:));
    hold off;
    ylabel("Electric field strength, E");
    xlabel("Position, x");
    
    
    subplot(1,5,3);
    title(['Density (t=' num2str(i-1) ')']);
    %ylim([0 10]);
    hold on;
    % plot(x_centre, electron_lo_dens_data(i,:));
    plot(x_centre, electron_lo_soln_data(i,1:nx),'r');
    plot(x_centre, electron_ho_dens_data(i,:),'b--');
    if ((nspecies >= 2) && (plot_species2 == 1))
        plot(x_centre, ion_lo_dens_data(i,:));
        plot(x_centre, ion_ho_dens_data(i,:));
    end
    hold off;
    if (nspecies >= 2)
        legend("LO-e", "HO-e", "LO-i", "HO-i");
    else
        legend("LO", "HO");
    end
    ylabel("Density, n");
    xlabel("Position, x");
    
    subplot(1,5,4);
    title(['Avg. mom (t=' num2str(i-1.5) ')']);
    hold on;
    if (i>1)
        % plot(x_face, electron_lo_avgmom_data(i,:), 'k*');
        plot(x_face(2:end), electron_lo_soln_data(i-1,nx+1:2*nx),'r');
        plot(x_face, electron_ho_avgmom_data(i-1,:),'b--');
        % plot(x_face, electron_ho_mom_data(i,:),'g*'); 
        if ((nspecies >= 2) && (plot_species2 == 1))
            plot(x_face, ion_lo_avgmom_data(i-1,:));
            plot(x_face, ion_ho_avgmom_data(i-1,:));
        end
        hold off;
        if ((nspecies >= 2) && (plot_species2 == 1))
            legend("LO-e", "HO-e", "LO-i", "HO-i");
        else
            legend("LO", "HO");
        end
        ylabel("Avg. momentum, nubar");
        xlabel("Position, x");
    end
    
    subplot(1,5,5);
    title(['Cont (t=' num2str(i-1.5) ')']);
    %ylim([0 10]);
    hold on;
    if (i > 1)
        % plot(x_centre, electron_lo_dens_data(i,:));
        plot(x_centre, electron_ho_cont_data(i-1,:),'b--');
        if ((nspecies >= 2) && (plot_species2 == 1))
            plot(x_centre, ion_ho_cont_data(i-1,:));
        end
        hold off;
        if ((nspecies >= 2) && (plot_species2 == 1))
            legend("electron", "ion");
        else
            legend("electron");
        end
        ylabel("Charge continuity residual");
        xlabel("Position, x");
    end
    
    pause(pause_time);
    
end
end