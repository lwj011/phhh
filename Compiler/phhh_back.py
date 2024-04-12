#this is the code of private hierarchical heavy hitters.
from Compiler import types, library, instructions
from Compiler.types import Array
import itertools
from Compiler.types import sint, cint

from Compiler.circuit import sha3_256
from Compiler.GC.types import sbitvec, sbits
from Compiler.instructions import time,crash



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
    @library.for_range(len(St_flat) - 1)
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
    #D.print_reveal_nested(end='\n')
    assert len(k) == len(D)  # k=[2,5,0,1]
    bs = types.Matrix.create_from(k.get_vector().bit_decompose(n_bits))  #bit_decompose bs=[[0,1,0,1],[1,0,0,0],[0,1,0,0],[0,0,0,0,],...[0,0,0,0]]
    if signed and len(bs) > 1:
        bs[-1][:] = bs[-1][:].bit_not()  # bs
    h = types.Array.create_from(types.sint(types.regint.inc(len(k))))  #sint Array [0,1,2,...,len(k)-1]
    @library.for_range(len(bs))  #
    def _(i):
        b = bs[i]
        c = gen_bit_perm(b)  #
        apply_perm(c, h, reverse=False)  # h1=c0(h0),h=cn...c3c2c1c0(h0),h0=[0,1,2,3,...]
        @library.if_e(i < len(bs) - 1)
        def _():
            apply_perm(h, bs[i + 1], reverse=True)
        @library.else_
        def _():
            #D.print_reveal_nested(end='\n')
            apply_perm(h, D, reverse=True)
            #D.print_reveal_nested(end='\n')
    D_order = inverse_permutation(h)  #
    #D.print_reveal_nested(end='\n')
    if get_D:
        return D
    else:
        return D_order

def phhh_1(k0,n_bits=16, t=1):
    # my first scheme for phhh, this scheme is secure and efficient. k0 is the data array, n_bits is the bit length. t is threshold.
    # leak: bit length n_bits, data number len(k0)
    k = k0.same_shape()
    k.assign(k0)
   
    
    sorted_data = radix_sort(k,k,n_bits,signed=False)
    #sorted_data.print_reveal_nested(end='\n')
    frequency = types.Array.create_from(types.sint(types.regint.inc(size=len(k),base=1,step=0)))#sint array [1,1,1...,1]
    equ = types.Matrix(n_bits,len(sorted_data)-1,sint)  #restore the equal information of k
    

    bs = types.Matrix.create_from(sorted_data.get_vector().bit_decompose(n_bits))  #bit_decompose
    #if len(bs) > 1:
        #bs[-1][:] = bs[-1][:].bit_not()  # bs
    bsh2l = bs.same_shape()
    bsh2l_t = types.Matrix(len(k0),n_bits,sint)
    @library.for_range(len(bs))
    def _(i):
        bsh2l[i] = bs[n_bits-1-i]
    @library.for_range(len(bsh2l))
    def _(i):
        @library.for_range(len(bsh2l[i]))
        def _(j):
            bsh2l_t[j][i] = bsh2l[i][j]
    #bs.print_reveal_nested(end='\n')
    
    b = bsh2l[0]
    @library.for_range(len(b)-1)
    def _(j):
        equ[0][j] = b[j].equal(b[j+1],1)
    @library.for_range(len(bsh2l)-1)
    def _(i):
        b = bsh2l[i+1] #bs[x], x must be a natural number
        @library.for_range(len(b)-1)
        def _(j):
            equ[i+1][j] = equ[i][j]*b[j].equal(b[j+1],1)
    #equ.print_reveal_nested(end='\n')
   
    #restore the hierarchical heavy hitters
    hdata = sint.Tensor([n_bits,len(k0),n_bits])

    @library.for_range(len(equ))
    def _(i1):
        i = len(equ)-i1-1    
        @library.for_range(len(k0)-1)
        def _(j1):
            j = len(k0) - j1 -1
            frequency[j-1] = frequency[j-1] + frequency[j]*equ[i][j-1]
            frequency[j] = frequency[j]*(1-equ[i][j-1])  # operations where one of the operands is an sint either result in an sint or an sinbit, the latter for comparisons
        @library.for_range(len(k0))
        def _(j2):
            temp_equ = frequency[j2].greater_equal(t)
            temp_bs = bsh2l_t[0].same_shape()
            temp_bs.assign(bsh2l_t[j2])
            @library.for_range(len(temp_bs))
            def _(l1):
                temp_bs[l1] = temp_bs[l1] * temp_equ +(-2)*(temp_equ - 1) # 2 represent null
            @library.for_range(len(temp_bs)-i-1)
            def _(l2):
                temp_bs[l2+i+1] = sint(2)
            hdata[i][j2].assign(temp_bs)
            frequency[j2] = frequency[j2]*(1-temp_equ)
    
    hdata.print_reveal_nested(end='\n')  #the true print
   
def compact(t,p1,p2):
    # t is 0 or 1, p1 and p2 is payload. compact t=1 items to the head.
    #leak:len(t)
    #t.print_reveal_nested(end='\n')
    c0 = p1.same_shape()
    c1 = p1.same_shape()
    label = p1.same_shape()
    c1[0] = t[0]
    c0[0] = sint(1) - c1[0]
    @library.for_range(len(t)-1)
    def _(i):
        c1[i+1] = c1[i] + t[i+1]
        c0[i+1] = sint(i+2) - c1[i+1]
    @library.for_range(len(t))
    def _(i):
        temp_equ = t[i].equal(sint(1))
        label[i] = c1[i]*temp_equ + (c0[i]+c1[len(t)-1])*(1-temp_equ) - 1
    #t.print_reveal_nested(end='\n')
    #label.print_reveal_nested(end='\n')
    apply_perm(label,t)
    apply_perm(label,p1)
    apply_perm(label,p2)
    return c1[len(t)-1]

def bit_equal(a, b, n_bits, get_bits):
    '''
    secretly compare the highest get_bits bits, a b is sint with n_bits, return sint 0/1
    leak:null
    '''

    dec_a =a.bit_decompose(n_bits)[n_bits-get_bits:]
    dec_b =b.bit_decompose(n_bits)[n_bits-get_bits:]
    bits = [1 - (bit_a - bit_b)%2 for bit_a,bit_b in zip(dec_a,dec_b)]
    while len(bits) > 1:
        bits.insert(0, bits.pop()*bits.pop())
    return bits[0]


def get_frequency(sorted_k0, n_bits, get_bits):
    # compute frequency, sorted_k0 is the sorted data, n_bits represents the length of data, get_bits represents calculating the frequency of the previous get_bits layers
    # return compacted data and frequency, the number of deduplication data c(cint)
    # leak:len(sorted_k0), the number of deduplication data
    k = sorted_k0.same_shape()
    k.assign(sorted_k0)
    indices = types.Array.create_from(types.sint(types.regint.inc(len(k))))  #sint Array [0,1,2,...,len(k)-1]
    equ = k.same_shape()
    equ.assign_all(0)
    equ[0] = sint(1)
    @library.for_range(len(k)-1)
    def _(i):
        equ[i+1] = sint(1) - bit_equal(k[i+1],k[i],n_bits, get_bits)
        #(k[i+1]-k[i]).reveal().print_reg_plain()
        #equ[i+1].reveal().print_reg_plain()
    #k.print_reveal_nested(end='\n')
    #equ.print_reveal_nested(end='\n')
    c1 = compact(equ,k,indices).reveal()  #leaking c1, the number of deduplication data
    frequency = indices.same_shape()
    frequency.assign_all(sint(0))
    @library.for_range(c1-1)  #surprise, for_range(start,stop,step) :param start/stop/step: regint/cint/int
    def _(i):
        frequency[i] = indices[i+1]-indices[i]
    frequency[c1-1] = len(indices) - indices[c1-1]
    return k, frequency, c1





class TrieNode:
    def __init__(self):
        self.data_count = sint(0)
        self.children = {}

class PrefixTree:
    def __init__(self):
        self.root = TrieNode()
    def insert(self, start_node, data, frequency):  #data is str, begin with the children of start_node
        current_node = start_node
        for bit in data:
            if bit not in current_node.children:
                current_node.children[bit] = TrieNode()
            current_node = current_node.children[bit]
        current_node.data_count += frequency
    def query(self, start_node, data):
        current_node = start_node
        for bit in data:
            if bit not in current_node.children:
                return sint(0)
        return current_node
 





def phhh_2(k0,n_bits=16, t=1):
    # my second scheme for phhh, this scheme is insecure and more efficient than phhh_1. k0 is the data array, n_bits is the bit length. t is threshold.
    # leak: bit length n_bits, data number len(k0),the number of deduplication data c
    k = k0.same_shape()
    k.assign(k0)
    
    sorted_data = radix_sort(k,k,n_bits,signed=False)
    #sorted_data.print_reveal_nested(end='\n')
    data, frequency, c = get_frequency(sorted_data, n_bits, n_bits)
    #data.print_reveal_nested(end='\n')
    #frequency.print_reveal_nested(end='\n')
    #c.print_reg_plain()
    @library.for_range(c, len(data))
    def _(i):
        data[i] *= sint(0)

    #data.assign_part_vector(sint(0),c)
    #data.print_reveal_nested(end='\n')
    
    tags = cint.Array(len(k))
    tags.assign_all(0)
    #tags.assign(0,c)
    #tags = [0 for _ in range(len(k))] # identify whether the data exists
    #tags = [1 if i < c else 0 for i in range(len(k))]
    #@library.for_range(c)
    #def _(i):
    #    tags[i] = 1
    
    #tags.print_reveal_nested(end='\n')
    


    bs = types.Matrix.create_from(data.get_vector().bit_decompose(n_bits))  #bit_decompose
    bsh2l = bs.same_shape()
    #bsh2l_t = types.Matrix(len(k0),n_bits,sint)
    @library.for_range(len(bs))
    def _(i):
        bsh2l[i] = bs[n_bits-1-i]
    #@library.for_range(len(bsh2l))
    #def _(i):
    #    @library.for_range(len(bsh2l[i]))
    #    def _(j):
    #        bsh2l_t[j][i] = bsh2l[i][j]
    #bsh2l.print_reveal_nested(end='\n')
    


    tree = PrefixTree()
    tags[0]
    @library.for_range(n_bits)
    def _(i):
        b = bs[i]
        b = b.reveal()

        





    tree.insert(tree.root,'0', sint(5))
    tree.insert(tree.root,'1', sint(12))


    

   
    '''

    bit_equal(sint(2), sint(1), 4,4).reveal().print_reg_plain() 
    bit_equal(sint(2), sint(1), 4,3).reveal().print_reg_plain()  
    bit_equal(sint(2), sint(1), 4,2).reveal().print_reg_plain()
    bit_equal(sint(2), sint(1), 4,1).reveal().print_reg_plain()
    bit_equal(sint(1), sint(2), 4,3).reveal().print_reg_plain()
    '''











    '''
    indices = types.Array.create_from(types.sint(types.regint.inc(len(k))))  #sint Array [0,1,2,...,len(k)-1]
    equ = k.same_shape()
    equ.assign_all(0)
    equ[0] = sint(1)
    @library.for_range(len(k)-1)
    def _(i):
        equ[i+1] = sorted_data[i+1].not_equal(sorted_data[i])
    c1 = compact(equ,sorted_data,indices).reveal()  #leaking c1, the number of deduplication data
    #equ.print_reveal_nested(end='\n')
    #sorted_data.print_reveal_nested(end='\n')
    #c1.reveal().print_reg_plain()
    #indices.print_reveal_nested(end='\n')
    frequency = indices.same_shape()
    frequency.assign_all(sint(0))
    @library.for_range(c1-1)  #surprise, for_range(start,stop,step) :param start/stop/step: regint/cint/int
    def _(i):
        frequency[i] = indices[i+1]-indices[i]
    frequency[c1-1] = len(indices) - indices[c1-1]
    #frequency.print_reveal_nested(end='\n')
    '''
    


