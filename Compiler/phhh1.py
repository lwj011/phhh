#this is the code of private hierarchical heavy hitters.

from Compiler import types, library, instructions
from Compiler.library import break_loop, start_timer, stop_timer
from Compiler.types import Array, MemValue
import itertools
from Compiler.types import sint, cint, regint, sintbit
from Compiler.circuit import sha3_256
from Compiler.GC.types import sbitvec, sbits
from Compiler.instructions import time,crash
import numpy as np
import csv
import random


def gen_bit_perm(b):
    """
    input:b=[0,1,0,1], output:[0,2,1,3]
    leak: null
    """
    B = types.sint.Matrix(len(b), 2)
    B.set_column(0, 1 - b.get_vector())
    B.set_column(1, b.get_vector())
    Bt = B.transpose()  #Bt=[[1,0,1,0],[0,1,0,1]]
    Bt_flat = Bt.get_vector()
    St_flat = Bt.value_type.Array(len(Bt_flat))
    St_flat.assign(Bt_flat)  #St_flat=[1,0,1,0,0,1,0,1]
    @library.for_range_opt(len(St_flat) - 1)
    def _(i):
        St_flat[i + 1] = St_flat[i + 1] + St_flat[i]
    Tt_flat = Bt.get_vector() * St_flat.get_vector()  # Tt_flat[[1],[0],[2],[0],[0],[3],[0],[4]]
    Tt = types.Matrix(*Bt.sizes, B.value_type)
    Tt.assign_vector(Tt_flat)  #Tt_flat  Tt=[[1,0,2,0], [0,3,0,4]]
      
    return sum(Tt) - 1  #[0,2,1,3]

def inverse_permutation(k):
    """
    get inverse permutation
    """
    shuffle = types.sint.get_secure_shuffle(len(k))  #shuffle
    k_prime = k.get_vector().secure_permute(shuffle).reveal()  #shuffle k,k_prime
    idx = Array.create_from(k_prime)
    res = Array.create_from(types.sint(types.regint.inc(len(k))))  #sint Array [0,1,2,...,len(k)-1]
    res.secure_permute(shuffle, reverse=False)  # shuffle res
    res.assign_slice_vector(idx, res.get_vector())
    library.break_point()
    instructions.delshuffle(shuffle)  #shuffle
    return res



def apply_perm(k, D, reverse=False):
    """
    apply k to D
    leak: null
    """
    assert len(k) == len(D)
    library.break_point()
    shuffle = types.sint.get_secure_shuffle(len(k))
    k_prime = k.get_vector().secure_permute(shuffle).reveal()
    idx = types.Array.create_from(k_prime)
    if reverse:
        D.assign_vector(D.get_slice_vector(idx))
        library.break_point()
        D.secure_permute(shuffle, reverse=True)
    else:
        D.secure_permute(shuffle)
        library.break_point()
        v = D.get_vector()
        D.assign_slice_vector(idx, v)
    library.break_point()
    instructions.delshuffle(shuffle)



def radix_sort(k0, D0, n_bits=16, get_D=True, signed=True):
    """
    this is same as the MP-SPDZ
    leak: only leak len(k)
    """
    k = k0.same_shape()
    k.assign(k0)
    D = D0.same_shape()
    D.assign(D0)
    assert len(k) == len(D)  # k=[2,5,0,1]
    bs = types.Matrix.create_from(k.get_vector().bit_decompose(n_bits))  #bit_decompose bs=[[0,1,0,1],[1,0,0,0],[0,1,0,0],[0,0,0,0,],...[0,0,0,0]]
    if signed and len(bs) > 1:
        bs[-1][:] = bs[-1][:].bit_not()  # bs
    h = types.Array.create_from(types.sint(types.regint.inc(len(k))))  #sint Array [0,1,2,...,len(k)-1]
    @library.for_range_opt(len(bs))  #
    def _(i):
        b = bs[i]
        c = gen_bit_perm(b)  #
        apply_perm(c, h, reverse=False)  # h1=c0(h0),h=cn...c3c2c1c0(h0),h0=[0,1,2,3,...]
        @library.if_e(i < len(bs) - 1)
        def _():
            apply_perm(h, bs[i + 1], reverse=True)
        @library.else_
        def _():
            apply_perm(h, D, reverse=True)
    D_order = inverse_permutation(h)  #
    if get_D:
        return D
    else:
        return D_order
    


    
def get_frequency_secure_phhh1(sum_freq, equ):
    # compute the frequency for phhh1; the function is euqal to frequency[j-1] = frequency[j-1] + frequency[j]*equ[i][j-1] for all j, which unable to parallelize
    # sum_freq[i] = freq[i] + freq[i+1] +...+ freq[n]
    # leak: len(sum_freq), len(equ)
    p = sum_freq.same_shape()
    p.assign(sum_freq)
    t = equ.same_shape()
    t.assign(equ)
    c0 = p.same_shape()
    c1 = p.same_shape()
    label = p.same_shape()

    #compact
    c1[0] = t[0]
    c0[0] = 1 - c1[0]
    @library.for_range_opt(len(t)-1)
    def _(i):
        c1[i+1] = c1[i] + t[i+1]
        c0[i+1] = i+2 - c1[i+1]
    label[:] = c1[:]*t[:] + (c0[:]+c1[len(t)-1])*(1-t[:]) - 1
    apply_perm(label,t)
    apply_perm(label,p)
  
    #get_frequency
    frequency = p.same_shape()
    frequency.assign_all(0)
    end = len(p)-1
    frequency[:end] = (t[:end]&t[1:]) * (p[:end] - p[1:]) + (t[:end]^t[1:]) * p[:end] 
    frequency[end] = t[end]
    apply_perm(label, frequency, reverse=True)
    return frequency




def phhh_1(k0,n_bits=16, t=1):
    # my first scheme for phhh, this scheme is secure and efficient. k0 is the data array, n_bits is the bit length. t is threshold.
    # leak: bit length n_bits, data number len(k0)

    #Define variables
    start_timer(10)
    start_timer(101)
    k = k0.same_shape()
    k.assign(k0)
    lenk = len(k0)
    lenk1 = len(k0) -1 
    start_timer(104)
    sorted_data = radix_sort(k,k,n_bits,signed=False)
    stop_timer(104)
    # sorted_data.print_reveal_nested(end=';')
    stop_timer(101)


    start_timer(102)
    frequency = types.Array.create_from(types.sint(types.regint.inc(size=len(k),base=1,step=0)))#sint array [1,1,1...,1]
    fres_t = types.Matrix(n_bits, len(k), sintbit)  # 1 represent is HHH
    equ = types.Matrix(n_bits, len(sorted_data), sintbit)  # 1 represent the first item for repeated data
    # start_timer(108)
    bs = types.Matrix.create_from(sorted_data.get_vector().bit_decompose(n_bits))  #bit_decompose
    # stop_timer(108)
    bsh2l = bs.same_shape()
    bsh2l_t = types.Matrix(len(k0),n_bits,sint)
    # start_timer(107)
    @library.for_range(len(bs))
    def _(i):
        bsh2l[i] = bs[n_bits-1-i]
    @library.for_range(len(bsh2l))
    def _(i):
        @library.for_range(len(bsh2l[i]))
        def _(j):
            bsh2l_t[j][i] = bsh2l[i][j]
    # stop_timer(107)

    #get equal
    start_timer(105)
    b = bsh2l[0] 
    equ[0][0] = 1
    equ[0][1:] = b[:lenk1]^b[1:]
    @library.for_range(n_bits -1)
    def _(i):
        b = bsh2l[i+1]
        equ[i+1][0] = 1
        equ[i+1][1:] = (b[:lenk1]^b[1:])|equ[i][1:]
    stop_timer(105)
    stop_timer(102)
    
    # get frequency
    start_timer(103)
    start_timer(106)
    @library.for_range(n_bits)
    def _(i1):
        i = n_bits-i1-1    
        @library.for_range(len(k0)-1)
        def _(j1):
            j = len(k0) -j1 -1
            frequency[j-1] = frequency[j-1] + frequency[j]
        frequency_temp = get_frequency_secure_phhh1(frequency, equ[i])
        frequency.assign(frequency_temp)
        fres_t[i][:] = frequency[:].greater_equal(t)
        frequency[:] = frequency[:]*(1-fres_t[i][:])
    

    # get HHH
    
    hdata_2 = bsh2l_t.same_shape() # due to memory limitations, I only generate and do not store. 2 represents the end of the data; it uses bhstl_t and fres_t to generate hdata
    hdata_2.assign(bsh2l_t)
    temp_null = hdata_2[0].same_shape()
    temp_null.assign_all(2)
    @library.for_range(n_bits-1)
    def _(i1):
        i = n_bits -1 -i1
        @library.for_range_parallel(1000, len(k0))
        def _(j):
            hdata_2[j][i] = 2
        hdata = hdata_2.same_shape() # due to memory limitations, I only generate and do not store. 2 represents the end of the data
        hdata.assign(hdata_2)
        @library.for_range_parallel(1000, len(k0))
        def _(j):
            hdata[j][:] = hdata[j][:] * fres_t[i-1][j] +2*(1 - fres_t[i-1][j])
            @library.if_(hdata[j][0].reveal()!=2)  #the output for observing
            def _():
                hdata[j].print_reveal_nested(end=',')
            # hdata[j].print_reveal_nested(end='; ')  #the true ouput without leaking
    hdata = bsh2l_t.same_shape() #for the lowest layer
    hdata.assign(bsh2l_t)  
    @library.for_range_parallel(1000, len(k0))
    def _(j):
        hdata[j][:] = hdata[j][:] * fres_t[n_bits-1][j] +2*(1 - fres_t[n_bits-1][j])
        @library.if_(hdata[j][0].reveal()!=2)  #the output for observing
        def _():
            hdata[j].print_reveal_nested(end=',')
        # hdata[j].print_reveal_nested(end='; ')  #the true ouput without leaking
    stop_timer(106)
    stop_timer(103)
    
    stop_timer(10)





   




def get_frequency_unsecure_phhh2(sorted_k0):
    # compute frequency, sorted_k0 is the sorted data, n_bits represents the length of data, get_bits represents calculating the frequency of the previous get_bits layers
    # return compacted data and frequency, the number of deduplication data c(cint)
    # leak:len(sorted_k0), the number of deduplication data c
    # this function is designed for PHHH2
    k = sorted_k0.same_shape()
    k.assign(sorted_k0)
    indices = types.Array.create_from(types.sint(types.regint.inc(len(k))))  #sint Array [0,1,2,...,len(k)-1]

    # get equal
    equ = sintbit.Array(len(k))
    equ.assign_all(0)
    equ[0] = sintbit(1)
    @library.for_range_parallel(500, len(k)-1)
    def _(i):
        equ[i+1] = (k[i+1].equal(k[i]))^1

    #compact
    p1 = k
    p2 = indices
    t = equ
    c0 = p2.same_shape()
    c1 = p2.same_shape()
    label = p2.same_shape()
    c1[0] = t[0]
    c0[0] = 1 - c1[0]
    @library.for_range_opt(len(t)-1)
    def _(i):
        c1[i+1] = c1[i] + t[i+1]
        c0[i+1] = i+2 - c1[i+1]
    label[:] = c1[:]*t[:] + (c0[:]+c1[len(t)-1])*(1-t[:]) - 1
    apply_perm(label,t)
    apply_perm(label,p1)
    apply_perm(label,p2)
    c = c1[len(t)-1].reveal() #leaking c, the number of deduplication data

    #get frequency
    frequency = indices.same_shape()
    frequency.assign_all(sint(0))
    @library.for_range_opt(c-1) 
    def _(i):
        frequency[i] = indices[i+1]-indices[i]
    frequency[c-1] = len(indices) - indices[c-1]
    return k, frequency, c





#I think mp-spdz do not support prefix tree
def phhh_2(k0,n_bits=16, t=1):
    # my second scheme for phhh, this scheme is insecure and more efficient than phhh_1. k0 is the data array, n_bits is the bit length. t is threshold.
    # leak: bit length n_bits, data number len(k0),the number of deduplication data c, b[s](except the node less than t), if nodeij>=t, shape of pruned prefixtree, the data corresponding to nodeij
    
    start_timer(20)

    start_timer(201)
    para_node = 80  #parallel parameter
    para_idx3 = 20
    
    
    k = k0.same_shape()
    k.assign(k0)
    start_timer(205)
    sorted_data = radix_sort(k,k,n_bits,signed=False) #sort
    stop_timer(205)
    stop_timer(201)
    start_timer(202)
    start_timer(206)
    data, frequency, c = get_frequency_unsecure_phhh2(sorted_data)  #get frequency of data, c is the number of duplicated data
    stop_timer(206)
    stop_timer(202)

    # define variables
    start_timer(203)
    tags = cint.Array(len(k)+1)  #1 is boundary point, 0 is useful, 2 is deleted
    tags.assign_all(0)
    tags[0] = cint(1) #1 is boundary point, 0 is useful, 2 is deleted, [1:1/2)
    tags.assign(2, len(k))
    idx = cint.Array(2*len(k)+1)  #Store the data range corresponding to valid nodes, idx [-1] represents the number of nodes * 2
    idx_p = sint.Array(2*len(k)+para_node)  #store the frequency for a layer, this is used for parallel
    idx_t = cint.Array(2*len(k)+para_node)  #store the relationship with t for a layer, this is used for parallel
    idx_div = cint.Array(len(k)+1)  #store the division point
    @library.for_range_parallel(500, len(data) - c)  #delete the duplicated data
    def _(i):
        data[i+c] = 0
        tags[i+c] = 2 
    bs = types.Matrix.create_from(data.get_vector().bit_decompose(n_bits))  #bit_decompose
    bsh2l = bs.same_shape()
    @library.for_range(len(bs))
    def _(i):
        bsh2l[i] = bs[n_bits-1-i]
    fres = types.Matrix(n_bits, len(k), sint) #store the frequencys of all layers
    fres.assign_all(0)
    fres_t = types.Matrix(n_bits, len(k), cint) # store the compare result, fres[i][j]>=t then fres_t[i][j]=1, else fres_t[i][j]=0, 2 represent do not know or no such node , 3 represent the origin is 1 but substract hhh items
    fres_t.assign_all(2)
    parent = types.Matrix(n_bits, len(k), cint) # restore the index of parent node, -1 represent null
    parent.assign_all(-1)
    

    
    #create the prefix tree
    start_timer(207)
    @library.for_range(n_bits)  # get the prefix tree with pruning, n_bits
    def _(i):
        b_reveal = cint.Array(len(bsh2l[i])+1) #Prevent overflow when searching for boundary points when all rows are 0
        b_reveal.assign(bsh2l[i].reveal())
        need_left = cint(1)  # 1 true; 0 false
        node_count = cint(0) # the 2*count of the node
        node_count_2 = cint(0) # the count of the node
        idx.assign_all(0)
        @library.for_range(c+1)  #find the left and right edge of the nodes, i.e. the idx
        def _(j):
            @library.if_((need_left==0).bit_and(tags[j]==1))  # can not change order
            def _():
                idx[node_count] = j-1  #right edge
                idx[node_count+1] = j  #left edge
                node_count.update(node_count + 2)
                node_count_2.update(node_count_2 + 1)
            @library.if_((need_left==0).bit_and(tags[j]==2))
            def _():
                idx[node_count] = j-1
                node_count.update(node_count + 1)
                need_left.update(1)
            @library.if_((need_left==1).bit_and(tags[j]==1))
            def _():
                idx[node_count] = j  #left edge
                node_count.update(node_count + 1)
                node_count_2.update(node_count_2 + 1)
                need_left.update(0)    
        idx[2*len(k)] = node_count
        
        idx_p.assign_all(0)
        idx_t.assign_all(0)
        idx_div.assign_all(0)
        @library.for_range(node_count_2)  #given a node, get the [frequency] information of the children node, node_count_2
        def _(j):
            left = idx[j*2]  #left edge
            right = idx[j*2+1]
            idx_div[j] = left  # Division point, belonging to the right side
            @library.while_do(lambda: (b_reveal[idx_div[j]]!=1).bit_and(idx_div[j]<(right+1)))  #find division point
            def _():
                idx_div[j] = idx_div[j] + 1
            @library.for_range(idx_div[j] -left)  #get frequency of the children nodes
            def _(s):
                idx_p[j*2] = idx_p[j*2] + frequency[s+left]
            @library.for_range(right - idx_div[j] + 1)
            def _(s):
                idx_p[j*2+1] = idx_p[j*2+1] + frequency[s+idx_div[j]]

        iter = node_count.max(para_node)  #para100 iter100 is faster than para100 iter50
        # node_count.print_reg_plain()
        @library.for_range_parallel(para_node, iter)  #given a node, get the [if>=t] information of the children node
        def _(j): 
            idx_t[j] = idx_p[j].greater_equal(t).reveal()
        


        @library.for_range(node_count_2)  #given a node, get the [pruning] information of the children node
        def _(j): 
            left = idx[j*2]
            right = idx[j*2+1]
            left_p = idx_p[j*2]  # left children node's frequency
            right_p = idx_p[j*2+1]
            left_t = idx_t[j*2]
            right_t = idx_t[j*2+1]
            div = idx_div[j]
            @library.if_(div==(right + 1))  #only have left children node
            def _():
                fres[i][left] = left_p  # the frequency of the left children node
                fres_t[i][left] = left_t
                parent[i][left] = left
                @library.if_(left_t==0)  #pruning
                def _():
                    @library.for_range(start=left,stop=right+1)
                    def _(s):
                        tags[s] = 2
                        @library.for_range(start=i+1,stop=n_bits)
                        def _(r):
                            bsh2l[r][s] = 2  #pruning to avoid leaking 
            @library.if_(div==left)  #only have right children node
            def _():
                fres[i][left] = right_p
                fres_t[i][left] = right_t
                parent[i][left] = left
                @library.if_(right_t==0)
                def _():
                    @library.for_range(start=left,stop=right+1)
                    def _(s):
                        tags[s] = 2
                        @library.for_range(start=i+1,stop=n_bits)
                        def _(r):
                            bsh2l[r][s] = 2  #pruning to avoid leaking
            @library.if_((div>left).bit_and(div<right+1))  #have two children nodes
            def _():
                fres[i][left] = left_p
                fres[i][div] = right_p
                fres_t[i][left] = left_t
                fres_t[i][div] = right_t
                parent[i][left] = left
                parent[i][div] = left
                tags[div] = 1
                @library.if_(left_t==0)
                def _():
                    @library.for_range(start=left,stop=div)
                    def _(s):
                        tags[s] = 2
                        @library.for_range(start=i+1,stop=n_bits)
                        def _(r):
                            bsh2l[r][s] = 2  #pruning to avoid leaking  
                @library.if_(right_t==0)
                def _():
                    @library.for_range(start=div,stop=right+1)
                    def _(s):
                        tags[s] = 2
                        @library.for_range(start=i+1,stop=n_bits)
                        def _(r):
                            bsh2l[r][s] = 2  #pruning to avoid leaking 
    stop_timer(207)
    stop_timer(203)        
    
    #get HHH items
    start_timer(204)
    start_timer(208)
    hhh = sint.Array(n_bits)
    idx_3 = cint.Array(len(k)+para_idx3)  # store the index of fre_t==3
    idx3_count = cint(0)
    @library.for_range(n_bits)
    def _(i):
        fre = fres[n_bits - i - 1]  # from the lowest layer to the highest layer
        fre_t = fres_t[n_bits -i -1]
        idx_3.assign_all(0)
        idx3_count.update(0)
        @library.for_range(c)  # get the relationship with t for the nodes changed
        def _(j):
            @library.if_(fre_t[j]==3)
            def _():
                idx_3[idx3_count] = j
                idx3_count.update(idx3_count+1)

        iter = idx3_count.max(para_idx3)
        # idx3_count.print_reg_plain()
        @library.for_range_parallel(para_idx3,iter)
        def _(j):
            fre_t[idx_3[j]] = fre[idx_3[j]].greater_equal(t).reveal()

        @library.for_range(c)
        def _(j):
            @library.if_((fre_t[j]==1))
            def _():
                hhh.assign_all(2)
                @library.for_range(n_bits - i)
                def _(s): 
                    hhh[s] = bsh2l[s][j]
                hhh.print_reveal_nested(end=',')  #the output for obversing& the true output, leaking the hhh value, which is hhh items
                temp_i = cint(0)
                temp_j = cint(0)
                temp_j.update(j)
                @library.for_range(n_bits - i -1)  # delete HHH items
                def _(s):
                    temp_i.update(n_bits -i - 1 - s - 1)
                    temp_j.update(parent[temp_i+1][temp_j])
                    fres[temp_i][temp_j] = fres[temp_i][temp_j] - fre[j]
                    fres_t[temp_i][temp_j] = 3
    stop_timer(208)
    stop_timer(204)
    
    stop_timer(20)



def bit_equal_sint_phhh0(a,b,n_bits,cmp_bits):
    """
    secretly compare the highest cmp_bits(cint,int) bits, a b is sint(n_bits(must be int)), return sintbit 0/1, equal return 1
    leak:null
    communication: 40 rounds
    """
    c = a>>(n_bits-cmp_bits)
    d = b>>(n_bits-cmp_bits)
    return c.equal(d)


def substract_phhh0(fress, datasij, datass, n_bits, s1, fresij, fres_tij):
    """
    communication: 50 rounds
    """
    fress[:] -= bit_equal_sint_phhh0(datasij, datass[:], n_bits, s1) * fresij * fres_tij
    return 0


 
def get_frequency_secure_phhh0(sorted_k0, n_bits, get_bits):
    # compute frequency, sorted_k0 is the sorted data, n_bits represents the length of data, get_bits represents calculating the frequency of the previous get_bits layers
    # return compacted data and frequency, the frequency of deleted data is 0
    # leak:len(sorted_k0)
    # reference: Vogue:Faster Computation of Private Heavy Hitters.
    # this function is designed for PHHH0(the trivial scheme)
    k = sorted_k0.same_shape()
    k.assign(sorted_k0)
    equ = sintbit.Array(len(k))
    equ.assign_all(0)
    indices = types.Array.create_from(types.sint(types.regint.inc(len(k))))  #sint Array [0,1,2,...,len(k)-1]

    #get equal
    equ[0] = 1
    @library.for_range_parallel(10, len(k)-1)  #get equ, 1 represents the first number of duplicate data
    def _(i):
        equ[i+1] = sintbit(1) - bit_equal_sint_phhh0(k[i+1],k[i],n_bits, get_bits)

    #compact
    p = indices
    t = equ
    c0 = p.same_shape()
    c1 = p.same_shape()
    label = p.same_shape()
    c1[0] = t[0]
    c0[0] = 1 - c1[0]
    @library.for_range_opt(len(t)-1)
    def _(i):
        c1[i+1] = c1[i] + t[i+1]
        c0[i+1] = i+2 - c1[i+1]
    label[:] = c1[:]*t[:] + (c0[:]+c1[len(t)-1])*(1-t[:]) - 1
    apply_perm(label,k)
    apply_perm(label,t)
    apply_perm(label,p)

    #get_frequency
    frequency = p.same_shape()
    frequency.assign_all(0)
    end = len(p)-1
    frequency[:end] = (t[:end]&t[1:]) * (p[1:] - p[:end]) + (t[:end]^t[1:]) * (len(k) - p[:end]) 
    frequency[end] = t[end]

    return k, frequency





def phhh_0(k0,n_bits=16, t=1):
    # the trivial scheme for phhh, this scheme is secure and  inefficient. k0 is the data array, n_bits is the bit length. t is threshold.
    # leak: bit length n_bits, data number len(k0)
    # round: 20l^2(lenk)^2/parallel  (if bit_equal is 40 rounds)  --> 50*n_bits*len(k)
    start_timer(30)
    k = k0.same_shape()
    k.assign(k0)
    start_timer(301)
    sorted_data = radix_sort(k,k,n_bits,signed=False)  
    stop_timer(301)
    fres = types.Matrix(n_bits, len(k0), sint)  #store frequency
    datas = types.Matrix(n_bits, len(k0), sint) #Store data corresponding to fres
    hdata = sint.Tensor([n_bits, len(k0), n_bits])  #store HHH
    fres_t = sintbit.Matrix(n_bits, len(k))  # 1 represent HHH
    fres.assign_all(0)
    hdata.assign_all(2)

    #get frequency and hdata
   
    @library.for_range(n_bits)  #get frequency and hdata
    def _(i_temp):
        i = n_bits - i_temp -1
        start_timer(302)
        datas[i], fres[i] = get_frequency_secure_phhh0(sorted_data, n_bits, n_bits - i_temp)  #get_frequency
        stop_timer(302)
        bs = types.Matrix.create_from(datas[i].get_vector().bit_decompose(n_bits))  #bit_decompose
        @library.for_range(len(k0))
        def _(j):
            @library.for_range(i+1)
            def _(s):
                hdata[i][j][s] = bs[n_bits - s - 1][j]
    
    
    #get HHH
    start_timer(303)
    @library.for_range(n_bits)
    def _(i_temp):
        i = n_bits - i_temp -1                    
        fres_t[i][:] = fres[i][:].greater_equal(t)
        start_timer(304)   
        @library.for_range(len(k))
        def _(j):
            res = Array.create_from(types.cint(types.regint.inc(size=n_bits, base=1)))  #cint Array [1,2,...,n_bits]
            @library.for_range_parallel(n_bits, n_bits)  #substract HHH items; Loop n_bits layers are beneficial for improving efficiency through parallelism without affecting accuracy, as the fres[i:] of additional modifications will not be used
            def _(s):
                substract_phhh0(fres[s], datas[i][j], datas[s], n_bits, res[s], fres[i][j], fres_t[i][j])
        stop_timer(304)
        @library.for_range(len(k)) #may parallel
        def _(j):
            hdata[i][j] = fres_t[i][j].if_else(hdata[i][j], sint(2))
    stop_timer(303)
    
    # hdata.print_reveal_nested(end='\n')  #the true output without leaking
    # @library.for_range_opt(len(hdata))   #the output for observing
    # def _(i):
    #     @library.for_range_opt(len(hdata[i]))
    #     def _(j):
    #         @library.if_(hdata[i][j][0].reveal()!=cint(2))
    #         def _():
    #             hdata[i][j].print_reveal_nested(end='; ')
    
    stop_timer(30)



def test_client_random(n_bits, num):
    start_timer(60)
    data = []
    for _ in range(num):
        # 生成一个随机数
        random_number = random.getrandbits(n_bits)
        data.append(random_number)
    a = sint.Array(num)
    a.assign(data)
    stop_timer(60)

def test_client_random16(num):
    n_bits=16
    start_timer(616)
    data = []
    for _ in range(num):
        # 生成一个随机数
        random_number = random.getrandbits(n_bits)
        data.append(random_number)
    a = sint.Array(num)
    a.assign(data)
    stop_timer(616)

def test_client_random32(num):
    n_bits=32
    start_timer(632)
    data = []
    for _ in range(num):
        # 生成一个随机数
        random_number = random.getrandbits(n_bits)
        data.append(random_number)
    a = sint.Array(num)
    a.assign(data)
    stop_timer(632)

def test_client_random48(num):
    n_bits = 48
    start_timer(648)
    data = []
    for _ in range(num):
        # 生成一个随机数
        random_number = random.getrandbits(n_bits)
        data.append(random_number)
    a = sint.Array(num)
    a.assign(data)
    stop_timer(648)

def test_client_random64(num):
    n_bits=64
    start_timer(664)
    data = []
    for _ in range(num):
        # 生成一个随机数
        random_number = random.getrandbits(n_bits)
        data.append(random_number)
    a = sint.Array(num)
    a.assign(data)
    stop_timer(664)





# The following content is the algorithm designed for the experiment

def generate_zipf_distribution(n_bits, num, zipf_exponent=1.03):
    max_value = 2**n_bits - 1
    sampled_points = []
    while len(sampled_points) < num:
        samples = np.random.zipf(zipf_exponent, size=num)
        samples = samples[samples <= max_value]
        sampled_points = np.concatenate((sampled_points, samples))
    sampled_points = sampled_points[:num]
    return list(map(int, sampled_points.astype(np.uint64)))
   


def generate_normal_distribution(n_bits, num, var=5):
    max_value = 2**n_bits - 1
    normal_data = np.random.normal(loc=max_value/2, scale=np.sqrt(var), size=num)
    normal_data = np.clip(normal_data, 0, max_value)
    return list(map(int, normal_data.astype(np.uint64)))



def generate_widemawi_distribution(n_bits, num):
    #n_bits=[16,32,48,64], num<=400k
    csv_file_path = './Compiler/wide_dataset/ip'+str(n_bits)+'_400k.csv'
    data_list = []
    count = 0
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if count >= num:
                break
            data_list.append(int(row[0]))
            count += 1

    return data_list




# get_frequency_secure_phhh1(sum_freq, equ):

def test_phhh1_frequency(k0, n_bits):
    """
    Test the cost of the frequency algorithm used in Phhh1 scheme
    """
    start_timer(60)
    start_timer(61)
    #sort
    k = k0.same_shape()
    k.assign(k0)
    sorted_data = radix_sort(k,k,n_bits,signed=False) #sort
    stop_timer(61)

    start_timer(64)
    start_timer(62)
    # get equal
    equ = sintbit.Array(len(sorted_data))
    equ.assign_all(0)
    equ[0] = sintbit(1)
    @library.for_range_parallel(500, len(sorted_data)-1)
    def _(i):
        equ[i+1] = (sorted_data[i+1].equal(sorted_data[i]))^1
    stop_timer(62)

    start_timer(63)
    #get frequency
    frequency = types.Array.create_from(types.sint(types.regint.inc(size=len(sorted_data),base=1,step=0)))#sint array [1,1,1...,1]
    @library.for_range(len(sorted_data)-1)
    def _(j1):
        j = len(sorted_data) -j1 -1
        frequency[j-1] = frequency[j-1] + frequency[j]
    frequency_temp = get_frequency_secure_phhh1(frequency, equ)
    stop_timer(63)
    stop_timer(64)


    # frequency_temp.print_reveal_nested(end='\n')
    stop_timer(60)

    
    

def test_phhh0_frequency(k0, n_bits):
    """
    Test the cost of the frequency algorithm used in Phhh0 scheme
    """
    start_timer(50)

    # k0.print_reveal_nested(end='\n')

    
    start_timer(51)
    #sort
    k = k0.same_shape()
    k.assign(k0)
    sorted_data = radix_sort(k,k,n_bits,signed=False) #sort
    stop_timer(51)

    start_timer(54)
    start_timer(52)
    # get equal
    equ = sintbit.Array(len(sorted_data))
    equ.assign_all(0)
    equ[0] = sintbit(1)
    @library.for_range_parallel(500, len(sorted_data)-1)
    def _(i):
        equ[i+1] = (sorted_data[i+1].equal(sorted_data[i]))^1
    stop_timer(52)

    start_timer(53)
    #compact
    indices = types.Array.create_from(types.sint(types.regint.inc(len(sorted_data))))  #sint Array [0,1,2,...,len(k)-1]
    p = indices
    t = equ
    c0 = p.same_shape()
    c1 = p.same_shape()
    label = p.same_shape()
    c1[0] = t[0]
    c0[0] = 1 - c1[0]
    @library.for_range_opt(len(t)-1)
    def _(i):
        c1[i+1] = c1[i] + t[i+1]
        c0[i+1] = i+2 - c1[i+1]
    label[:] = c1[:]*t[:] + (c0[:]+c1[len(t)-1])*(1-t[:]) - 1
    apply_perm(label,sorted_data)
    apply_perm(label,t)
    apply_perm(label,p)

    #get_frequency
    frequency = p.same_shape()
    frequency.assign_all(0)
    end = len(p)-1
    frequency[:end] = (t[:end]&t[1:]) * (p[1:] - p[:end]) + (t[:end]^t[1:]) * (len(sorted_data) - p[:end]) 
    frequency[end] = t[end]
    stop_timer(53)
    stop_timer(54)


    # sorted_data.print_reveal_nested(end='\n')
    # frequency.print_reveal_nested(end='\n')
    stop_timer(50)

    


    
    






