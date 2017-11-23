import os
from cosmosis.datablock import names, option_section
import sys

#add class directory to the path
dirname = os.path.split(__file__)[0]
#enable debugging from the same directory
if not dirname.strip(): dirname='.'
install_dir = dirname+"/hi_class_devel/classy_install/lib/python2.7/site-packages/"
sys.path.insert(0, install_dir)

import classy
import numpy as np

#These are pre-defined strings we use as datablock
#section names
cosmo = names.cosmological_parameters
distances = names.distances
cmb_cl = names.cmb_cl

def setup(options):
    #Read options from the ini file which are fixed across
    #the length of the chain
    config = {
        'lmax': options.get_int(option_section,'lmax', default=2000),
        'zmax': options.get_double(option_section,'zmax', default=4.0),
        'kmax': options.get_double(option_section,'kmax', default=50.0),
    }

    #Create the object that connects to Class
    config['cosmo'] = classy.Class()

    #Return all this config information
    return config

def get_class_inputs(block, config):

    #Get parameters from block and give them the
    #names and form that class expects
    smgs = smg_params(block)
    try:
        block[cosmo,'omega_smg']
    except NameError:
        params = {
            'output': 'tCl,pCl,lCl,mPk',
            'modes': 's, t',
            'l_max_scalars': config["lmax"],
            'P_k_max_h/Mpc':  config["kmax"],
            'z_pk': ', '.join(str(z) for z in np.arange(0.0, config['zmax'], 0.01)),
            'lensing': 'no',
            'A_s':       block[cosmo, 'A_s'],
            'n_s':       block[cosmo, 'n_s'],
            'H0':        100*block[cosmo, 'h0'],
            'omega_b':   block[cosmo, 'ombh2'],
            'omega_cdm': block[cosmo, 'omch2'],
            'tau_reio':  block[cosmo, 'tau'],
            'T_cmb':     block.get_double(cosmo, 't_cmb', default=2.726),
            'N_eff':     block.get_double(cosmo, 'massless_nu', default=3.046),
            'r':         block.get_double(cosmo, 'r_t', default=0.),
            #'Omega_smg': block.get_double(cosmo, 'omega_smg', default = 0.),
        }
    else:
        if block[cosmo,'omega_smg'] < 0.:
            params = {
                'output': 'tCl,pCl,lCl,mPk',
                'modes': 's, t',
                'expansion_model':'lcdm',
                'gravity_model':'propto_omega',
                'l_max_scalars': config["lmax"],
                'P_k_max_h/Mpc':  config["kmax"],
                'z_pk': ', '.join(str(z) for z in np.arange(0.0, config['zmax'], 0.01)),
                'lensing': 'no',
                'A_s':       block[cosmo, 'A_s'],
                'n_s':       block[cosmo, 'n_s'],
                'H0':        100*block[cosmo, 'h0'],
                'omega_b':   block[cosmo, 'ombh2'],
                'omega_cdm': block[cosmo, 'omch2'],
                'tau_reio':  block[cosmo, 'tau'],
                'T_cmb':     block.get_double(cosmo, 't_cmb', default=2.726),
                'N_eff':     block.get_double(cosmo, 'massless_nu', default=3.046),
                'r':         block.get_double(cosmo, 'r_t', default=0.),
                'Omega_Lambda': block[cosmo, 'omega_Lambda'],
                'Omega_smg': block.get_double(cosmo, 'omega_smg', default = 0.),
                'Omega_fld': block.get_double(cosmo, 'omega_fld', default = 0.),
                #'w0_fld': block.get_double(cosmo, 'w0_fld', default = -0.9),
                #'wa_fld': block[cosmo, 'wa_fld'],
                #'cs2_fld': block[cosmo, 'cs2_fld'],
                #'Omega_scf': block.get_double(cosmo, 'omega_scf', default = 0.),
                'expansion_smg':block.get_double(cosmo, 'expansion_smg', default = 0.5),
                'parameters_smg':block.get_string(cosmo, 'parameters_smg', default = smgs),
            }
        else:
            params = {
                'output': 'tCl,pCl,lCl,mPk',
                'modes': 's, t',
                'l_max_scalars': config["lmax"],
                'P_k_max_h/Mpc':  config["kmax"],
                'z_pk': ', '.join(str(z) for z in np.arange(0.0, config['zmax'], 0.01)),
                'lensing': 'no',
                'A_s':       block[cosmo, 'A_s'],
                'n_s':       block[cosmo, 'n_s'],
                'H0':        100*block[cosmo, 'h0'],
                'omega_b':   block[cosmo, 'ombh2'],
                'omega_cdm': block[cosmo, 'omch2'],
                'tau_reio':  block[cosmo, 'tau'],
                'T_cmb':     block.get_double(cosmo, 't_cmb', default=2.726),
                'N_eff':     block.get_double(cosmo, 'massless_nu', default=3.046),
                'r':         block.get_double(cosmo, 'r_t', default=0.),
                'Omega_smg': block.get_double(cosmo, 'omega_smg', default = 0.),
            }
    
        
    return params


def get_class_outputs(block, c, config):
    ##
    ## Derived cosmological parameters
    ##

    block[cosmo, 'sigma_8'] = c.sigma8()
    h0 = block[cosmo, 'h0']

    ##
    ##  Matter power spectrum
    ##

    #Ranges of the redshift and matter power
    dz = 0.01
    kmin = 1e-4
    kmax = config['kmax']*h0
    nk = 100

    #Define k,z we want to sample
    z = np.arange(0.0, config["zmax"]+dz, dz)
    k = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    nz = len(z)

    #Extract (interpolate) P(k,z) at the requested
    #sample points.
    P = np.zeros((nk, nz))
    for i,ki in enumerate(k):
        for j,zj in enumerate(z):
            P[i,j] = c.pk(ki,zj)

    #Save matter power as a grid
    block.put_grid("matter_power_lin", "k_h", k/h0, "z", z, "p_k", P*h0**3)

    ##
    ##Distances and related quantities
    ##

    #save redshifts of samples
    block[distances, 'z'] = z
    block[distances, 'nz'] = nz

    #Save distance samples
    block[distances, 'd_l'] = np.array([c.luminosity_distance(zi) for zi in z])
    d_a = np.array([c.angular_distance(zi) for zi in z])
    block[distances, 'd_a'] = d_a
    block[distances, 'd_m'] = d_a * (1+z)
    block[distances, 'h'] = np.array([c.Hubble(zi) for zi in z])

    #Save some auxiliary related parameters
    block[distances, 'age'] = c.age()
    block[distances, 'rs_zdrag'] = c.rs_drag()

    ##
    ## Now the CMB C_ell
    ##
    c_ell_data =  c.raw_cl()
    ell = c_ell_data['ell']
    ell = ell[2:]

    #Save the ell range
    block[cmb_cl, "ell"] = ell

    #t_cmb is in K, convert to mu_K, and add ell(ell+1) factor
    tcmb_muk = block[cosmo, 't_cmb'] * 1e6
    f = ell*(ell+1.0) / 2 / np.pi * tcmb_muk**2

    #Save each of the four spectra
    for s in ['tt','ee','te','bb']:
        block[cmb_cl, s] = c_ell_data[s][2:] * f

def execute(block, config):
    c = config['cosmo']

    #Set input parameters
    params = get_class_inputs(block, config)
    c.set(params)

    #Run calculations
    c.compute()

    #Extract outputs
    get_class_outputs(block, c, config)

    #Reset for re-use next time
    c.struct_cleanup()
    return 0

def cleanup(config):
    config['cosmo'].empty()

def smg_params(block): # il = initial parameters_smg list, ep = position of the element you wish to chnnge, ne = value f the new element
   # sl = list(parameters_smg)
   # for n in range(len(sl)):
   #     if sl[n]==',':
   #         sl[n]=' '

    #sn = "".join(sl)

    #snl=map(float, sn.split()) # the list of floats

    #snl[ep]=ne # change the value of the element
    smg1 = block.get_double(cosmo, 'parameters_smg__1', default = 1.)
    smg2 = block.get_double(cosmo, 'parameters_smg__2', default = 0.)
    smg3 = block.get_double(cosmo, 'parameters_smg__3', default = 0.)
    smg4 = block.get_double(cosmo, 'parameters_smg__4', default = 0.)
    smg5 = block.get_double(cosmo, 'parameters_smg__5', default = 1.)
    snl = [smg1,smg2,smg3,smg4,smg5]

    #sn1 = ",".join(map(str, snl))  # convert back to string
    smg = ",".join(map(str, snl))
    return smg
