import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Attention, MultiHeadAttention, Flatten, Reshape, Concatenate, RepeatVector
from tensorflow import math
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

#get substrate nodes' flexibilities
def get_nf(R, r):
    rf = r.flatten()
    Rrep = np.repeat(R, repeats=rf.shape[0])
    rrep = np.tile(rf, reps=R.shape[0])
    mask = np.ones([rrep.shape[0]])
    mask[rrep<=0] = 0
    allocable = (Rrep >= rrep) * mask
    return allocable.reshape(R.shape[0],rf.shape[0]).sum(axis=1)

#get vnfs' flexibilities
def get_vf(R, r):
    rf = r.flatten()
    Rrep = np.tile(R, reps=rf.shape[0])
    rrep = np.repeat(rf, repeats=R.shape[0])
    mask = np.ones([rrep.shape[0]])
    mask[rrep<=0] = 0
    allocable = (rrep <= Rrep) * mask
    return allocable.reshape(rf.shape[0],R.shape[0]).sum(axis=1).reshape(r.shape)

#allocate single vnf to single substrate node
def allocate(R,r,si,ri,Ri):
    Rt = R.copy()
    rt = r.copy()
    res = rt[si,ri]
    Rt[Ri] -= res
    rt[si,ri] -= res
    return Rt, rt

#allocate vnfs in single slice to substrate nodes
def allocate_slice(R, r, si, al_ed, unal):
    s = r[si]
    slice_mapping = {}
    Rt = R.copy()
    rt = r.copy()
    allocated = al_ed.copy()
    unallocable = unal.copy()

    #iterate through vnfs in slice in descending demands
    for vi in np.argsort(-s):
        if unallocable[si,vi]==1:
            continue
        Rb = Rt.copy()
        rb = rt.copy()
        max_nf = 0
        max_vf = 0
        alc = False
        for ni in range(len(R)):
            if Rt[ni] < rt[si,vi]:
                continue
            Rti, rti = allocate(Rt,rt,si,vi,ni)
            cur_nf = get_nf(Rti,rti).sum()
            allocated_tmp = allocated.copy()
            allocated_tmp[si,vi] = 1
            if ((1 - allocated_tmp > 0) & (unallocable == 0)).sum() > 0:
                cur_vf = get_vf(Rti,rti)[(1 - allocated_tmp > 0) & (unallocable == 0)].min()
            else:
                cur_vf = max_vf
    #         print(vi, ni, max_nf, min_vf, cur_nf, cur_vf)
            if cur_vf >= max_vf:
                if cur_nf >= max_nf:
                    Rb = Rti
                    rb = rti
                    max_vf = cur_vf
                    max_nf = cur_nf
                    alc = True
                    slice_mapping[(si,vi)] = ni
    #                 print('allocated')
        if alc:
            Rt = Rb
            rt = rb
            allocated[si,vi] = 1
            unallocable[(get_vf(Rti,rti) == 0) & (allocated == 0)] = 1
    return Rt, rt, allocated, unallocable, slice_mapping

#allocate slices by descending total demands
def allocate_slices_tot_dem(R, r):
    r_ts = r.sum(axis=1)
    Rt = R.copy()
    rt = r.copy()
    allocated = np.zeros(r.shape)
    unallocable = np.zeros(r.shape)
    slice_mapping = {}

    for si in np.argsort(-r_ts):
        if unallocable[si].sum() != 0:
            continue
        Rt, rt, allocated, unallocable, s0 = allocate_slice(Rt, rt, si, allocated, unallocable)
        slice_mapping.update(s0)
    
    return Rt, rt, allocated, unallocable, slice_mapping

#allocate slices by descending total demands
def allocate_slices_max_dem(R, r):
    r_ts = r.max(axis=1)
    Rt = R.copy()
    rt = r.copy()
    allocated = np.zeros(r.shape)
    unallocable = np.zeros(r.shape)
    slice_mapping = {}

    for si in np.argsort(-r_ts):
        if unallocable[si].sum() != 0:
            continue
        Rt, rt, allocated, unallocable, s0 = allocate_slice(Rt, rt, si, allocated, unallocable)
        slice_mapping.update(s0)
    
    return Rt, rt, allocated, unallocable, slice_mapping

#allocate slices by descending total demands
def allocate_slices_min_dem(R, r):
    r_ts = r.min(axis=1)
    Rt = R.copy()
    rt = r.copy()
    allocated = np.zeros(r.shape)
    unallocable = np.zeros(r.shape)
    slice_mapping = {}

    for si in np.argsort(-r_ts):
        if unallocable[si].sum() != 0:
            continue
        Rt, rt, allocated, unallocable, s0 = allocate_slice(Rt, rt, si, allocated, unallocable)
        slice_mapping.update(s0)
    
    return Rt, rt, allocated, unallocable, slice_mapping

#get total allocated resources to all slices
def total_allocated_res(R, r, slice_mapping):
    tmp = np.zeros(R.shape)
    for key in slice_mapping:
        tmp[slice_mapping[key]] += r[key[0],key[1]]
    return tmp

def allocate_all_vnfs(R, r):
    s_indx = np.vstack(np.unravel_index(np.argsort(-rq, axis=None), rq.shape)).T
    slice_mapping = {}
    Rt = R.copy()
    rt = r.copy()
    allocated = np.zeros(rq.shape)
    unallocable = np.zeros(rq.shape)
    #iterate through vnfs in all slices in descending demands
    for si,vi in s_indx:
        if unallocable[si].sum() != 0:
            continue
        Rb = Rt.copy()
        rb = rt.copy()
        max_nf = 0
        max_vf = 0
        alc = False
        for ni in range(len(R)):
            if Rt[ni] < rt[si,vi]:
                continue
            Rti, rti = allocate(Rt,rt,si,vi,ni)
            cur_nf = get_nf(Rti,rti).sum()
            allocated_tmp = allocated.copy()
            allocated_tmp[si,vi] = 1
            if ((1 - allocated_tmp > 0) & (unallocable == 0)).sum() > 0:
                cur_vf = get_vf(Rti,rti)[(1 - allocated_tmp > 0) & (unallocable == 0)].min()
            else:
                cur_vf = max_vf
            if cur_vf >= max_vf:
                if cur_nf >= max_nf:
                    Rb = Rti
                    rb = rti
                    max_vf = cur_vf
                    max_nf = cur_nf
                    alc = True
                    slice_mapping[(si,vi)] = ni
        if alc:
            Rt = Rb
            rt = rb
            allocated[si,vi] = 1
            unallocable[(get_vf(Rti,rti) == 0) & (allocated == 0)] = 1
    return Rt, rt, allocated, unallocable, slice_mapping

#allocating vnfs using deep agent
def agent_allocate(R, r, model):
    Rt = R.copy()
    rt = r.copy()
    allocated_slices = []
    unallocable_slices = set()
    allocated = np.zeros(rq.shape)
    unallocable = np.zeros(rq.shape)
    slice_mapping = {}

    while len(allocated_slices) + len(unallocable_slices) < ns:
        mask = np.ones((1,ns))
        sorted_si = np.argsort(-model.predict([Rt.reshape(1,-1), rt.reshape(1,ns,nv), mask])[0].flatten())
        for si in sorted_si:
            if (si in allocated_slices) or (si in unallocable_slices):
                continue
            Rt, rt, allocated, unallocable, s0 = allocate_slice(Rt, rt, si, allocated, unallocable)
            slice_mapping.update(s0)
            allocated_slices.append(si)
            unallocable_slices.update(np.argwhere(unallocable.sum(axis=1)!=0).flatten())
            break
    return allocated_slices, slice_mapping

import functools
import operator

def gen_mask(a, in_shape, window):
    msks = np.zeros([in_shape, window])
    msks[np.arange(in_shape),a] = 1
    return msks

def gen_y(Qs, msks):
    y = np.array(Qs)
    return msks * y[:,None]

def slice_eps(l, ix):
    a = [l[i] for i in ix]
    return functools.reduce(operator.iconcat, a, [])