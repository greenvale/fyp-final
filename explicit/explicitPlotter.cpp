clear
clc
close all

disp_particles = 0;
nspecies = 1;
nx = 40;
lx = 1.0;
dx = lx/nx;
numParticles = 10000;
nt = 20000 / 1;

x_face = linspace(0,lx,nx+1);

%% DATA IMPORT

if (disp_particles == 1)
    electron_pos_file = fopen('output/electron_pos.txt','r');
    electron_pos_data = fscanf(electron_pos_file, '%f');
    fclose(electron_pos_file);

    electron_vel_file = fopen('output/electron_vel.txt','r');
    electron_vel_data = fscanf(electron_vel_file, '%f');
    fclose(electron_vel_file);

    electron_pos_data = reshape(electron_pos_data, numParticles,[]);
    electron_pos_data = electron_pos_data';

    electron_vel_data = reshape(electron_vel_data, numParticles,[]);
    electron_vel_data = electron_vel_data';
end

elec_file = fopen('output/elec.txt','r');
elec_data = fscanf(elec_file, '%f');
fclose(elec_file);

elec_data = reshape(elec_data, nx+1,[]);
elec_data = elec_data';

electron_dens_file = fopen('output/electron_dens.txt','r');
electron_dens_data = fscanf(electron_dens_file, '%f');
fclose(electron_dens_file);

electron_dens_data = reshape(electron_dens_data, nx+1,[]);
electron_dens_data = electron_dens_data';

electron_mom_file = fopen('output/electron_mom.txt','r');
electron_mom_data = fscanf(electron_mom_file, '%f');
fclose(electron_mom_file);

electron_mom_data = reshape(electron_mom_data, nx+1,[]);
electron_mom_data = electron_mom_data';

if nspecies >= 2
    if (disp_particles == 1)
        ion_pos_file = fopen('output/ion_pos.txt','r');
        ion_pos_data = fscanf(ion_pos_file, '%f');
        fclose(ion_pos_file);

        ion_vel_file = fopen('output/ion_vel.txt','r');
        ion_vel_data = fscanf(ion_vel_file, '%f');
        fclose(ion_vel_file);

        ion_pos_data = reshape(ion_pos_data, numParticles,[]);
        ion_pos_data = ion_pos_data';

        ion_vel_data = reshape(ion_vel_data, numParticles,[]);
        ion_vel_data = ion_vel_data';
    end

    ion_dens_file = fopen('output/ion_dens.txt','r');
    ion_dens_data = fscanf(ion_dens_file, '%f');
    fclose(ion_dens_file);

    ion_dens_data = reshape(ion_dens_data, nx+1,[]);
    ion_dens_data = ion_dens_data';

    ion_mom_file = fopen('output/ion_mom.txt','r');
    ion_mom_data = fscanf(ion_mom_file, '%f');
    fclose(ion_mom_file);

    ion_mom_data = reshape(ion_mom_data, nx+1,[]);
    ion_mom_data = ion_mom_data';
end

%% FIGURE 1

figure(1);
set(gcf,'posit',[8,14,1800,400]);
for i=1:nt
    clf;
    
    subplot(1,4,1);
    hold on;
    title('Position vs velocity');
    if disp_particles == 1
    plot(electron_pos_data(i,:), electron_vel_data(i,:), 'b.');
    if (nspecies >= 2)
        plot(ion_pos_data(i,:), ion_vel_data(i,:), 'r.');
    end
    hold off;
    end
    
    subplot(1,4,2);
    ylim([-1e-9, 1e-9]);
    title('Electric field');
    hold on;
    plot(x_face, elec_data(i,:));
    hold off;
    
    subplot(1,4,3);
    title('Density');
    ylim([0 2]);
    hold on;
    plot(x_face, electron_dens_data(i,:));
    if (nspecies >= 2)
        plot(x_face, ion_dens_data(i,:));
    end
    hold off;
    if (nspecies >= 2)
        legend("e", "i");
    end
    
    subplot(1,4,4);
    title('Momentum');
    hold on;
    plot(x_face, electron_mom_data(i,:));
    if (nspecies >= 2)
        plot(x_face, ion_mom_data(i,:));
    end
    hold off;
    if (nspecies >= 2)
        legend("e", "i");
    end
    
    pause(.01);
    
end