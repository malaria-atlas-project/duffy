# from mcmc import *
from model import *
from cut_geographic import cut_geographic, hemisphere
import duffy
from postproc_utils import *
import pymc as pm
from pymc import thread_partition_array
import numpy as np
import os
root = os.path.split(duffy.__file__)[0]
pm.gp.cov_funs.cov_utils.mod_search_path.append(root)

cut_matern = pm.gp.cov_utils.covariance_wrapper('matern', 'pymc.gp.cov_funs.isotropic_cov_funs', {'diff_degree': 'The degree of differentiability of realizations.'}, 'cut_geographic', 'cg')
 
nugget_labels = {'sp_sub_b': 'V_b', 'sp_sub_0': 'V_0'}
obs_labels = {'sp_sub_b':'eps_p_fb','sp_sub_0':'eps_p_f0'}

def check_data(ra):
    pass

def phe0(sp_sub_b, sp_sub_0, p1):
    cmin, cmax = thread_partition_array(sp_sub_b)
    out = sp_sub_b.copy('F')     
    pm.map_noreturn(phe0_postproc, [(out, sp_sub_0, p1, cmin[i], cmax[i]) for i in xrange(len(cmax))])
    return out

def gena(sp_sub_b, sp_sub_0, p1):
    cmin, cmax = thread_partition_array(sp_sub_b)        
    out = sp_sub_b.copy('F')         
    pm.map_noreturn(gena_postproc, [(out, sp_sub_0, p1, cmin[i], cmax[i]) for i in xrange(len(cmax))])
    return out
    
def genb(sp_sub_b, sp_sub_0):
    cmin, cmax = thread_partition_array(sp_sub_b)        
    out = sp_sub_b.copy('F')         
    pm.map_noreturn(genb_postproc, [(out, sp_sub_0, cmin[i], cmax[i]) for i in xrange(len(cmax))])
    return out
    
def gen0(sp_sub_b, sp_sub_0):
    cmin, cmax = thread_partition_array(sp_sub_b)        
    out = sp_sub_b.copy('F')         
    pm.map_noreturn(gen0_postproc, [(out, sp_sub_0, cmin[i], cmax[i]) for i in xrange(len(cmax))])
    return out
    
map_postproc = [phe0, gena, genb, gen0]

def phe0(data):
    
    if not np.all((data.datatype=='prom')+(data.datatype=='gen')+(data.datatype=='phe')):
        raise ValueError, 'Hold-out dataset contains non-\'phe0-compatible\' points.'
    
    
    where_gen = np.where(data.datatype=='gen')
    where_phe = np.where(data.datatype=='phe')
    where_prom = np.where(data.datatype=='prom')
    
    obs = np.empty(len(data), dtype='float')
    obs[where_phe] = data[where_phe].phe0
    obs[where_prom] = data[where_prom].prom0
    obs[where_gen] = (np.sum([(data[k][where_gen]) for k in ['gen00','gen01','gen11']], axis=0))
    
    def f(sp_sub_b, sp_sub_0, p1):
        p = sp_sub_b.copy('F')
        phe0_postproc(p, sp_sub_0, p1)
        pred = pm.rbinomial(n=data.n, p=p)
        
        return pred
    return obs, data.n, f

def het_ab(data):
    if not np.all(data.datatype=='gen'):
        raise ValueError, 'Hold-out dataset contains non-\'gen\' points.'
    
    # Observed number of heterozygotes at the a/b locus
    obs = (data.genab+data.gena0+data.genb1+data.gen01).astype('float')
    def f(sp_sub_b, sp_sub_0, p1):
        from model import g_freqs

        # Probability of drawing a chromosome with 'a', not 'b', regardless of silencing mutations.
        pa = [.5*np.sum([g_freqs[k](spb, sp0, p1) for k in ['ab','a0','b1','01']])\
            + np.sum([g_freqs[k](spb, sp0, p1) for k in ['aa','a1','11']])\
                for spb, sp0 in zip(pm.flib.invlogit(sp_sub_b), pm.flib.invlogit(sp_sub_0))]

        pa = np.asarray(pa).ravel()

        # Draw from predictive distribution of number of heterozygotes at the a/b locus
        pred = pm.rbinomial(n=data.n, p=2*pa*(1-pa))
        if pred.shape != sp_sub_b.shape:
            raise ValueError
        
        return pred
    return obs, data.n, f
    
def het_0(data):
    if not np.all(data.datatype=='gen'):
        raise ValueError, 'Hold-out dataset contains non-\'gen\' points.'
    
    # Observed number of heterozygotes at the silencing locus
    obs = data.genb0.astype('float')
    def f(sp_sub_b, sp_sub_0, p1):
        from model import g_freqs

        # Probability of drawing a chromosome with the silencing mutation, given that it is a 'b'.
        p0 = [[.5*g_freqs['b0'](spb, sp0, p1) + g_freqs['00'](spb, sp0, p1)] \
            for spb, sp0 in zip(pm.flib.invlogit(sp_sub_b), pm.flib.invlogit(sp_sub_0))]
        p0 = np.asarray(p0).ravel()

        # Draw from predictive distribution of number of heterozygotes at the silencing locus
        pred = pm.rbinomial(n=data.n, p=2*p0*(1-p0))
        if pred.shape != sp_sub_b.shape:
            raise ValueError

        return pred
    return obs, data.n, f
        
# Uncomment to choose validation metric.        
validate_postproc = [phe0]
# validate_postproc = [het_ab, het_0]
    
metadata_keys = []

def mcmc_init(M):
    M.use_step_method(pm.gp.GPEvaluationGibbs, M.sp_sub_b, M.V_b, M.eps_p_fb_d)
    M.use_step_method(pm.gp.GPEvaluationGibbs, M.sp_sub_0, M.V_0, M.eps_p_f0_d)
    for tup in zip(M.eps_p_fb_d, M.eps_p_f0_d):
        M.use_step_method(pm.AdaptiveMetropolis, tup)
        # for v in tup:
        #     M.use_step_method(pm.Metropolis, v)
    scalar_stochastics = []
    for v in M.stochastics:
        if v not in M.eps_p_fb_d and v not in M.eps_p_f0_d and np.alen(v.value) == 1 and v.dtype != np.dtype('object'):
            scalar_stochastics.append(v)
    M.use_step_method(pm.gp.GPParentAdaptiveMetropolis, scalar_stochastics, scales = dict([(s,.01*np.asscalar(s.value)) for s in scalar_stochastics]))


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