# from mcmc import *
from model import *
from generic_mbg import FieldStepper, thread_partition_array
from cut_geographic import cut_geographic, hemisphere
import duffy
from postproc_utils import duffy_postproc
import pymc as pm
import numpy as np
import os
root = os.path.split(duffy.__file__)[0]
pm.gp.cov_funs.cov_utils.mod_search_path.append(root)

# Stuff mandated by the new map_utils standard
cut_matern = pm.gp.cov_utils.covariance_wrapper('matern', 'pymc.gp.cov_funs.isotropic_cov_funs', {'diff_degree': 'The degree of differentiability of realizations.'}, 'cut_geographic', 'cg')
 
f_labels = ['eps_p_fb', 'eps_p_f0']
fs_have_nugget = {'eps_p_fb': True, 'eps_p_f0': True}
nugget_labels = {'eps_p_fb': 'V_b', 'eps_p_f0': 'V_0'}
M_labels = {'eps_p_fb': 'M_b', 'eps_p_f0': 'M_0'}
C_labels = {'eps_p_fb': 'C_b', 'eps_p_f0': 'C_0'}
x_labels = {'eps_p_fb': 'data_mesh', 'eps_p_f0': 'data_mesh'}
diags_safe = {'eps_p_fb': True, 'eps_p_f0': True}

def phe0(eps_p_fb, eps_p_f0, p1):
    """
    Returns probability of Duffy negativity from two random fields giving mutation frequencies.
    Fast and threaded.
    """
    cmin, cmax = thread_partition_array(eps_p_fb)        
    pm.map_noreturn(duffy_postproc, [(eps_p_fb, eps_p_f0, p1, cmin[i], cmax[i]) for i in xrange(len(cmax))])
    return eps_p_fb

def gena(eps_p_fb, eps_p_f0, p1):
    return (1-eps_p_fb)*p1
    
def genb(eps_p_fb, eps_p_f0):
    return eps_p_fb*(1-eps_p_f0)
    
def gen0(eps_p_fb, eps_p_f0):
    return eps_p_fb * eps_p_f0
    
map_postproc = [phe0, gena, genb, gen0]

def validate_postproc(**non_cov_columns):
    """
    Don't know what to do here yet.
    """
    raise NotImplementedError
    
metadata_keys = ['fi','ti','ui']

def mcmc_init(M):
    M.use_step_method(FieldStepper, M.fb, M.V_b, M.C_eval_b, M.M_eval_b, M.logp_mesh, M.eps_p_fb, M.ti)
    M.use_step_method(FieldStepper, M.f0, M.V_0, M.C_eval_0, M.M_eval_0, M.logp_mesh, M.eps_p_f0, M.ti)
    for tup in zip(M.eps_p_fb_d, M.eps_p_f0_d):
        M.use_step_method(pm.AdaptiveMetropolis, tup)
        # for v in tup:
        #     M.use_step_method(pm.Metropolis, v)
    scalar_stochastics = []
    for v in M.stochastics:
        if v not in M.eps_p_fb_d and v not in M.eps_p_f0_d and np.squeeze(v.value).shape == ():
            scalar_stochastics.append(v)
            M.use_step_method(pm.Metropolis, v)
    M.use_step_method(pm.AdaptiveMetropolis, scalar_stochastics)
            

non_cov_columns = { 'n': 'int',
                    'datatype': 'str',
                    'genaa': 'float',
                    'genab': 'float',
                    'genbb': 'float',
                    'gen00': 'float',
                    'gena0': 'float',
                    'genb0': 'float',
                    'gena1': 'float',
                    'genb1': 'float',
                    'gen01': 'float',
                    'gen11': 'float',
                    'pheab': 'float',
                    'phea': 'float',
                    'pheb': 'float',
                    'phe0': 'float',
                    'prom0': 'float',
                    'promab': 'float',
                    'aphea': 'float',
                    'aphe0': 'float',
                    'bpheb': 'float',
                    'bphe0': 'float'}
