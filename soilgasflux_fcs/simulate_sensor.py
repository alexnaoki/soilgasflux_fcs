import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from soilgasflux_fcs import models

class Simulate_Sensor:
    def __init__(self, alpha, cs, t0, c0, total_time, dt, temperature=20, pressure=101325):
        self.alpha = alpha
        self.cs = cs
        self.t0 = t0
        self.c0 = c0
        self.total_time = total_time
        self.dt = dt
        self.time = np.arange(0, total_time, dt)
        self.dcdt = None
        self.c = None
        self.temperature = temperature
        self.pressure = pressure

        # self.elements = {'volumes': [], 'concentrations': []}
        self.elements = {}

    def chamber_settings(self, area, chamber_volume):
        '''
        area: [cm2]
        volume: [cm3]
        '''
        self.area = area
        self.volume = chamber_volume

        self.chamber_nodes = int(chamber_volume/area)
        gas_mass = np.zeros((self.total_time, self.chamber_nodes))*np.nan
        inital_massConcentration=self._idealgaslaw_convertion_to_massConcentration(self.c0, molar_mass=44.01, 
                                                          temperature=self.temperature, 
                                                          pressure=self.pressure)#/self.chamber_nodes
        gas_mass[0, :] = inital_massConcentration*self.volume/self.chamber_nodes #per node

        # self.elements['volumes'].append({'chamber': [chamber_volume]})
        # self.elements['concentrations'].append({'chamber': [self.c0]})
        self.elements['chamber'] = {'volume': [chamber_volume], 'area': [area],
                                    'concentration': [self.c0],
                                    'gas_mass': gas_mass}

        # print('Created chamber with area:', area, 'cm2 and volume:', chamber_volume, 'cm3')
        print(f'Chamber settings: area={area:.3f} cm2, volume={chamber_volume:.3f} cm3')
        # print('Initial concentration:', self.c0, 'ppm')
    
    def gasAnalyzer_settings(self, response_time, gasAnalyzer_volume, sensor_accuracy, sensor_precision):
        '''
        response_time: [s]
        gasAnalyzer_volume: [cm3]
        sensor_accuracy: [ppm]
        sensor_precision: [ppm]
        '''
        self.response_time = response_time
        self.gasAnalyzer_volume = gasAnalyzer_volume
        self.sensor_accuracy = sensor_accuracy
        self.sensor_precision = sensor_precision
        gas_mass = np.zeros((self.total_time, 1))*np.nan
        inital_massConcentration=self._idealgaslaw_convertion_to_massConcentration(self.c0, molar_mass=44.01, 
                                                          temperature=self.temperature, 
                                                          pressure=self.pressure)
        gas_mass[0] = inital_massConcentration*self.gasAnalyzer_volume

        self.elements['gas_analyzer'] = {'volume': [gasAnalyzer_volume], 
                                         'concentration':[self.c0],
                                         'gas_mass': gas_mass}
        # self.elements['volumes'].append({'gas_analyzer': [gasAnalyzer_volume]})
        # self.elements['concentrations'].append({'gas_analyzer': [self.c0]})

        print(f'Gas Analyzer settings: response_time={response_time:.3f} s, volume={gasAnalyzer_volume:.3f} cm3, accuracy={sensor_accuracy:.3f} ppm, precision={sensor_precision:.3f} ppm')
    
    def internal_pump_settings(self, pump_volume, pump_rate):
        '''
        pump_volume: [cm3]
        pump_rate: [cm3/s]
        '''
        self.pump_volume = pump_volume
        self.pump_rate = pump_rate

        gas_mass = np.zeros((self.total_time, 1))*np.nan
        inital_massConcentration=self._idealgaslaw_convertion_to_massConcentration(self.c0, molar_mass=44.01,
                                                          temperature=self.temperature, 
                                                          pressure=self.pressure)
        gas_mass[0] = inital_massConcentration*self.pump_volume

        # self.elements['volumes'].append({'internal_pump': [pump_volume]})
        # self.elements['concentrations'].append({'internal_pump': [self.c0]})
        self.elements['internal_pump'] = {'volume': [pump_volume], 
                                          'concentration':[self.c0],
                                          'gas_mass':gas_mass}

        print(f'Internal Pump settings: volume={pump_volume:.3f} cm3, rate={pump_rate:.3f} cm3/s')

    def additional_settings(self, volume):
        '''
        volume: [cm3]
        '''
        self.additional_volume = volume
        gas_mass = np.zeros((self.total_time, 1))*np.nan
        inital_massConcentration=self._idealgaslaw_convertion_to_massConcentration(self.c0, molar_mass=44.01,
                                                          temperature=self.temperature, 
                                                          pressure=self.pressure)
        gas_mass[0] = inital_massConcentration*self.additional_volume
        # self.elements['volumes'].append({'additional': [volume]})
        # self.elements['concentrations'].append({'additional': [self.c0]})
        self.elements['additional'] = {'volume': [volume], 
                                       'concentration':[self.c0],
                                       'gas_mass': gas_mass}
        print(f'Additional settings: volume={volume:.3f} cm3')

    def sim_ideal_curve(self):
        self.cx = models.hm_model(t=self.time, cx=self.cs, a=self.alpha, t0=self.t0, c0=self.c0)
        self.dcdt = models.hm_model_dcdt(t0=self.t0, c0=self.c0, a=self.alpha, cx=self.cs, t=self.time)

        fig, ax = plt.subplots(1,2, figsize=(6, 3))
        ax[0].scatter(self.time, self.cx,s=2)
        ax[0].set_title('Ideal Curve')
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('Concentration [ppm]')   
        ax[1].plot(self.time, self.dcdt)
        ax[1].set_title('Ideal dcdt')
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('dcdt [ppm/s]')
        fig.tight_layout()
        fig.show()

    def _idealgaslaw_convertion_to_massConcentration(self, ppm, molar_mass, temperature, pressure):
        '''
        ppm = concentration in ppm
        molar_mass = [g/mol]
        temperature = [Celsius]
        pressure = [Pa]

        output:
        mass_concentration = [g/cm3]
        '''
        R = 8.3145*10e6 #cm3 Pa K-1 mol-1
        mass_concentration = (pressure*ppm/10e6 * molar_mass) / (R * (temperature + 273.15)) # g/cm3
        return mass_concentration

    def _idealgaslaw_convertion_to_ppm(self, mass_concentration, molar_mass, temperature, pressure):
        '''
        mass_concentration = [g/cm3]
        molar_mass = [g/mol]
        temperature = [Celsius]
        pressure = [Pa]
        '''
        R = 8.3145*10e6 #cm3 Pa K-1 mol-1
        ppm = (mass_concentration * R * (temperature + 273.15)) / (pressure * molar_mass) * 10e6 # ppm
        return ppm

    

    def run_simulation(self):
        # initial condition
        #build elements
            # chamber
            # gas analyzer
            # internal pump
        # main loop
            # gas analyzer measure
            # add gas from soil to chamber
            # mix inside chamber
            # pump gas out of chamber to the gas analyzer
            # pump gas from gas analyzer to the chamber

        print('')
        for element in self.elements:
            print(f'Element: {element}')
            for key, value in self.elements[element].items():
                if key == 'gas_mass':
                    continue
                print(f'\t{key}: {value}')
        
        total_volume = 0
        for element in self.elements:
            total_volume += self.elements[element]['volume'][0]
        print('Total volume:', total_volume, 'cm3')

        gasDensity_sourceRate = self._idealgaslaw_convertion_to_massConcentration(ppm=1, 
                                                              molar_mass=44.01, 
                                                              temperature=20, 
                                                              pressure=101325) # g/cm3
        gasMass_sourceRate = gasDensity_sourceRate * total_volume
        print('Gas density source rate:\t', gasDensity_sourceRate, 'g/cm3')
        print('gas mass source rate:\t', gasMass_sourceRate, 'g/s')
        print()

        # print('gas mass before loop',self.elements['chamber']['gas_mass'][0,:], 'g')


        time = 0
        # print(self.time)
        for t in self.time:
            if t == self.time.max():
                # continue
                print('End of simulation')
                break
                # pass
                
            print('Time:\t',t)

            # Measure gas
            gas_analyzer_mass = self.elements['gas_analyzer']['gas_mass'][t]
            gas_analyzer_volume = self.elements['gas_analyzer']['volume'][0]
            gas_analyzer_concentration = self._idealgaslaw_convertion_to_ppm(gas_analyzer_mass/gas_analyzer_volume,
                                                                           molar_mass=44.01, 
                                                                           temperature=self.temperature, 
                                                                           pressure=self.pressure)
            print('Gas analyzer mass:', gas_analyzer_mass, 'g')
            print('Gas analyzer volume:', gas_analyzer_volume, 'cm3')
            print('Gas analyzer concentration:', gas_analyzer_concentration, 'ppm')
            
            
            # Chamber input source
            gasMass_source = gasMass_sourceRate * self.dt
            print('gas mass source:', gasMass_source, 'g')
            # self.elements['chamber']['gas_mass'][t,:] = self.elements['chamber']['gas_mass'][t-1,:]
            self.elements['chamber']['gas_mass'][t,0] += gasMass_source
            
            print('Initail gas mass after input')
            print(self.elements['chamber']['gas_mass'][t,:], 'g')
            # Chamber diffusion
            if True:
                stocastic_diffusion = np.random.uniform(low=0, high=1, size=(self.chamber_nodes-1))
                for n in range(1,self.chamber_nodes,1):
                    proportion_diffusion = stocastic_diffusion[n-1]
                    node_diff_split = abs(self.elements['chamber']['gas_mass'][t, n-1] - self.elements['chamber']['gas_mass'][t, n])/2

                    if self.elements['chamber']['gas_mass'][t, n-1] > self.elements['chamber']['gas_mass'][t, n]:
                        self.elements['chamber']['gas_mass'][t, n-1] -= node_diff_split*proportion_diffusion
                        self.elements['chamber']['gas_mass'][t, n] += node_diff_split*proportion_diffusion
                    else:
                        self.elements['chamber']['gas_mass'][t, n-1] += node_diff_split*proportion_diffusion
                        self.elements['chamber']['gas_mass'][t, n] -= node_diff_split*proportion_diffusion
            
            # Pumping from chamber to gas analyzer
            pumped_volume = 0
            pumped_gasmass = 0
            print('\nPUMPING parts')
            for parts in self.elements:
                if parts == 'chamber':
                    continue
                pumped_volume += self.elements[parts]['volume'][0]

                self.elements[parts]['gas_mass'][t+1] = 0
                # print(self.elements[parts]['volume'][0])
                # print(self.elements[parts]['gas_mass'][t], 'g')
                print(self._idealgaslaw_convertion_to_ppm(self.elements[parts]['gas_mass'][t]/self.elements[parts]['volume'][0],
                                                          molar_mass=44.01, 
                                                          temperature=self.temperature, 
                                                          pressure=self.pressure), 'ppm')
                
                pumped_gasmass += self.elements[parts]['gas_mass'][t]
                print(f'gas mass pumped from {parts}',self.elements[parts]['gas_mass'][t], 'g')
                # print(self.elements[parts]['gas_mass'][0])
                if pumped_volume < self.pump_rate*self.dt:
                    print('More volume to be pumped', pumped_volume,'/', self.pump_rate*self.dt)
                else:
                    continue

            # print('Initial gas mass chamber')
            # print(self.elements['chamber']['gas_mass'][t], 'g')

            # To be pumped from chamber
            if pumped_volume < self.pump_rate*self.dt:
                print('chamber pump')
                need_to_pump = self.pump_rate*self.dt - pumped_volume

                chamber_nodes_volume = np.ones(self.chamber_nodes)*self.elements['chamber']['volume'][0]/self.chamber_nodes

                # Initialize arrays to track pumping
                volume_pumped_from_nodes = np.zeros(self.chamber_nodes)
                mass_pumped_from_nodes = np.zeros(self.chamber_nodes)
                
                
                self.elements['chamber']['gas_mass'][t+1, :] = self.elements['chamber']['gas_mass'][t, :]
                # Start from the last node (typically closest to the outlet)
                for n in range(self.chamber_nodes-1, -1, -1):
                    if need_to_pump <= 0:
                        break

                    # Calculate how much to pump from this node
                    volume_to_pump = min(chamber_nodes_volume[n], need_to_pump)
                    volume_pumped_from_nodes[n] = volume_to_pump
                    need_to_pump -= volume_to_pump

                    # Calculate fraction of node volume being pumped
                    fraction_pumped = volume_to_pump / chamber_nodes_volume[n]
                    
                    # Calculate gas mass to be pumped (proportional to volume)
                    mass_to_pump = self.elements['chamber']['gas_mass'][t, n] * fraction_pumped
                    # print('node mass', mass_to_pump, self.elements['chamber']['gas_mass'][t])

                    mass_pumped_from_nodes[n] = mass_to_pump

                    self.elements['chamber']['gas_mass'][t+1, n] -= mass_to_pump
                
                # print(volume_pumped_from_nodes)
                print(mass_pumped_from_nodes.sum())
                print('mass nodes pumped',mass_pumped_from_nodes)
                # pumped_gasmass += np.nansum(mass_pumped_from_nodes)
                pumped_gasmass += mass_pumped_from_nodes.sum()
                pumped_volume += np.nansum(volume_pumped_from_nodes)
                print('Pumped gas mass:', pumped_gasmass, 'g')

                print()
            
            pumped_gasDensity_backtoChamber = pumped_gasmass / pumped_volume # g/cm3
            print('pumped gas density',pumped_gasDensity_backtoChamber, 'g/cm3')
            print('mass pumped', pumped_gasmass, 'g')
            print('volume pumped', pumped_volume, 'cm3')

            # Move gas from chamber to rest of the system and from lower nodes from the chamber
            print('\n moving...')


            moved_volume = 0
            need_to_move = self.pump_rate*self.dt
            # moved_mass = 0
            mass_to_move = 0
            
            volume_moved_from_nodes = np.zeros(self.chamber_nodes)
            # print('before moved nodes', self.elements['chamber']['gas_mass'][t+1, :], 'g')
            for n in range(self.chamber_nodes-1, -1, -1):
                print('moved:', moved_volume)
                # print(chamber_nodes_volume)
                if moved_volume >= need_to_move:
                    break

                # Calculate how much to move from this node
                volume_to_move = min(chamber_nodes_volume[n]-volume_pumped_from_nodes[n], need_to_move-moved_volume)
                volume_moved_from_nodes[n] = volume_to_move
                moved_volume += volume_to_move

                # Calculate fraction of node volume being moved
                fraction_moved = volume_to_move / (chamber_nodes_volume[n]-volume_pumped_from_nodes[n])
                # print(fraction_moved)

                # Calculate gas mass to be moved (proportional to volume)
                print('adding',self.elements['chamber']['gas_mass'][t+1, n]*fraction_moved, 'g')
                mass_to_move += self.elements['chamber']['gas_mass'][t+1, n] * fraction_moved

                self.elements['chamber']['gas_mass'][t+1, n] -= self.elements['chamber']['gas_mass'][t+1, n] * fraction_moved

            # print('moved nodes', volume_moved_from_nodes)
            # print('after moved nodes', self.elements['chamber']['gas_mass'][t+1, :], 'g')

            moved_gasDensity = mass_to_move / moved_volume # g/cm3
            print('moved gas density:', moved_gasDensity, 'g/cm3')
            print('mass moved',mass_to_move, 'g')
            print('volume moved', moved_volume, 'cm3')

            # Replenish the chamber with gas from the rest of the system
            # left_moved_volume = need_to_move - moved_volume
            for parts in self.elements:
                if parts == 'chamber':
                    continue
                print(self.elements[parts]['gas_mass'][t+1], 'g')
                self.elements[parts]['gas_mass'][t+1] += moved_gasDensity*self.elements[parts]['volume'][0]
                print(self.elements[parts]['gas_mass'][t+1], 'g (new)')
                
            # print('chamber gas mass before moving', self.elements['chamber']['gas_mass'][t+1, :], 'g')
            self.elements['chamber']['gas_mass'][t+1, :] += moved_gasDensity*volume_pumped_from_nodes
            # print('chamber gas mass after moving',self.elements['chamber']['gas_mass'][t+1, :], 'g')

            ## Second move ##
            # print((self.elements['chamber']['volume'][0]/self.chamber_nodes-volume_moved_from_nodes))
            gasMass_concentration_to_move = self.elements['chamber']['gas_mass'][t+1, :]/(self.elements['chamber']['volume'][0]/self.chamber_nodes-volume_moved_from_nodes)
            print('concentration to move',gasMass_concentration_to_move)

            # print(volume_moved_from_nodes)
            # print(self.elements['chamber']['volume'][0]/self.chamber_nodes-volume_moved_from_nodes)

            # initial state of chamber volume nodes... needs to update in this loop
            chamberMove_nodes_volume = self.elements['chamber']['volume'][0]/self.chamber_nodes-volume_moved_from_nodes
            print(chamberMove_nodes_volume)
            for n in range(self.chamber_nodes-1,-1,-1):
                print()
                # if the chamber already has the right amount of gas, skip
                if chamberMove_nodes_volume[n] == self.elements['chamber']['volume'][0]/self.chamber_nodes:
                    continue
                # if it is  the first node, stop
                elif n == 0:
                    break

                # print(f'Iteartion node {n}:',chamberMove_nodes_volume)
                # print('total voume:', np.sum(chamberMove_nodes_volume))
                # print('node mass:', self.elements['chamber']['gas_mass'][t+1, :])

                # go through the nodes before the current node to move gas mass/volume to the current until it is full
                n_1 = n - 1
                while chamberMove_nodes_volume[n] < self.elements['chamber']['volume'][0]/self.chamber_nodes:
                    print(n_1)
                    node_n_volume_needed = self.elements['chamber']['volume'][0]/self.chamber_nodes-chamberMove_nodes_volume[n]

                    if chamberMove_nodes_volume[n] == self.elements['chamber']['volume'][0]/self.chamber_nodes:
                        break
                    
                    else:
                        if node_n_volume_needed >= chamberMove_nodes_volume[n_1]:
                            # add mass from node n_1 to node n until node n is full or node n_1 is empty
                            self.elements['chamber']['gas_mass'][t+1, n] += self.elements['chamber']['gas_mass'][t+1, n_1] 
                            # substract mass and volume from n_1
                            self.elements['chamber']['gas_mass'][t+1, n_1] = 0
                            chamberMove_nodes_volume[n_1] = 0
                            # update volume of node n
                            chamberMove_nodes_volume[n] = self.elements['chamber']['volume'][0]/self.chamber_nodes
                        
                        elif node_n_volume_needed < chamberMove_nodes_volume[n_1]:
                            # add mass from node n_1 to node n until node n is full or node n_1 is empty
                            self.elements['chamber']['gas_mass'][t+1, n] += self.elements['chamber']['gas_mass'][t+1, n_1] * (node_n_volume_needed/chamberMove_nodes_volume[n_1])
                            # substract mass and volume from n_1
                            self.elements['chamber']['gas_mass'][t+1, n_1] -= self.elements['chamber']['gas_mass'][t+1, n_1] * (node_n_volume_needed/chamberMove_nodes_volume[n_1])
                            chamberMove_nodes_volume[n_1] -= node_n_volume_needed
                            chamberMove_nodes_volume[n] = self.elements['chamber']['volume'][0]/self.chamber_nodes
                    n_1 -= 1

                # print(f'after Iteartion node {n}:',chamberMove_nodes_volume)
                # print('total voume:', np.sum(chamberMove_nodes_volume))
                # print('node mass:', self.elements['chamber']['gas_mass'][t+1, :])
                


            ### Return volume and mass to the chamber ###
            print('chamber node volume', chamberMove_nodes_volume)
            print()
            volume_to_add_from_pumped = self.elements['chamber']['volume'][0]/self.chamber_nodes-chamberMove_nodes_volume
            print('volume to add from pumped', volume_to_add_from_pumped)
            # for n in range(self.chamber_nodes):
            self.elements['chamber']['gas_mass'][t+1,n] += pumped_gasDensity_backtoChamber*volume_to_add_from_pumped[n]






            print()

        fig, ax =plt.subplots(2,3, figsize=(10, 6))
        for t in self.time:
            # print(t/self.time)
            ax[0,0].plot(self.elements['chamber']['gas_mass'][t,:], alpha=t/self.time.max())
        
        gas_mass_system = self.elements['chamber']['gas_mass'][:,:].sum(axis=1)+self.elements['gas_analyzer']['gas_mass'][:]+self.elements['internal_pump']['gas_mass'][:]+self.elements['additional']['gas_mass'][:]

        # print(gas_mass_system)

        chamber_gasMass = self.elements['chamber']['gas_mass'].sum(axis=1)+self.elements['gas_analyzer']['gas_mass'].reshape(-1)
        print(chamber_gasMass)
        # print

        ax[0,1].plot(self.time, chamber_gasMass, 'blue')
        # ax[1].axhline(gasMass_sourceRate, color='r', linestyle='')
        # ax[0,1].plot(gasMass_sourceRate*self.time, 'r--')
        # print(gas_mass_system)
        # total_gasMass = 
        
        # for t in self.time:
        #     ax[0,2].scatter(t, self.elements['chamber']['gas_mass'][t,:].sum(), s=2, 
        #                     # alpha=t/self.time.max() 
        #                     )
            
        for parts in self.elements:
            # if cha
            ax[0,2].plot(self.elements[parts]['gas_mass'][:].sum(axis=1)/self.elements[parts]['volume'][0], label=parts)
            ax[0,2].legend()





    
