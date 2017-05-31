import copy
import numpy as np
import itertools

def createAllTuple_(symbols, dim, tuples):
    if( dim > 0 ):
        tmpTuples = []
        for tup in tuples:
            for s in symbols:
                tmpTuples.append(tup + [s])
        uniqTuples = []
        for tup in tmpTuples:
            found = 0
            for tup2 in uniqTuples:
                if( tup2 == tup ):
                    found = 1
                    break
            if( found == 0 ):
                uniqTuples.append(tup)
        return createAllTuple_(symbols, dim-1, uniqTuples)
    else:
        return tuples

def createAllTupple(symbols, dim):
    """ generate all possible dim-tuples with all the symbols of 'symbols'
        e.g., createAllTupple([a,b],3) will return [[a,b,b],[a,a,b],[a,b,a],[b,a,a],[b,a,b],[b,b,a]]
        this function is used for the diagonal blocks
    """
    tuples = [[]]
    tuples = createAllTuple_(symbols, dim, tuples)
    ret = []
    # remove all those tuples which do not use all the provided symbols
    for tup in tuples:
        okay = 1
        for s in symbols:
            found = 0
            for t in tup:
                if(t == s):
                    found = 1
            if( found == 0 ):
                okay = 0
        if( okay ):
            ret.append(tup)
    return ret


def getInversePerm(perm):
    invPerm = []
    for i in range(len(perm)):
        invPerm.append(perm.index(i))
    return tuple(invPerm)


def applyPerm(perm, index):
    permutedIndex = []
    for i in range(len(index)):
        permutedIndex.append(index[perm[i]])
    return tuple(permutedIndex)

def arrayToString(array, clean = 0):
    ret = ""
    for a in array:
        ret += "%s"%a
        if( clean != 1 ):
            ret += ","
    if( clean != 1 ):
        return ret[0:-1]
    else:
        return ret

def hasItem( L, item ):
    for l in L:
        if( l == item):
            return 1
    return 0

def unrollPermutations(statements):
    # unrolls : T_abc = A_abc + A_acb
    #           B_abc = T_abc + T_cba
    # To:       B_abc = A_abc + A_acb + A_cba + A_bca
    naiveSummation = copy.deepcopy(statements[0])
    for l in range(1,len(statements)):
        tmp = []
        for (scalar2, perm2) in naiveSummation:
            for (scalar1, perm1) in statements[l]:
                tmp.append((scalar1*scalar2,applyPerm(perm1,perm2)))
        naiveSummation = tmp
    return naiveSummation 

def getTrashCache(floatType):
    ret = ""
    ret += "void trashCache(%s *A, %s *B, int n){\n"%(floatType, floatType)
    ret += "#pragma omp parallel for\n"
    ret += "for(int i = 0; i < n; i++ )\n"
    ret += "A[i] += 0.999 * B[i];\n"
    ret += "}\n"
    return ret

def dumpToFile(metrics, filename):
    transposed = np.asarray(metrics).T.tolist()

    counter = 0
    content = ""
    for measurement in transposed:
        line = "%d"%counter
        for metric in measurement:
            line += " %.2f"%(metric)
        content += line + "\n"
        counter += 1
    if( content != "" ):
        fp = open(filename,"w")
        fp.write(content)
        fp.close()

def solvePermutationLeft(permLhs, permRhs):
    """Solves permX o permLhs = permRhs for 'permX'"""
    permX = [ -1 for i in permLhs]
    for i in range(len(permLhs)):
        permX[i] = permLhs.index(permRhs[i])

    #vvvvv DEBUG vvvvv
    if( not arraysEqual(applyPerm(permX,applyPerm(permLhs, range(len(permLhs)))), permRhs) ):
        print permX, "o", permLhs,"=",permRhs
        print "ERROR in solvePermutationLeft"
        exit(-1)
    return permX

def solvePermutationRight(permLhs, permRhs):
    """Solves permLhs o permX = permRhs for 'permX'"""
    permX = [ -1 for i in permLhs]
    for i in range(len(permLhs)):
        permX[permLhs[i]] = permRhs[i] 

    #vvvvv DEBUG vvvvv
    if( not arraysEqual(applyPerm(permLhs,applyPerm(permX, range(len(permLhs)))), permRhs) ):
        print permLhs, "o", permX ,"=",permRhs
        print "ERROR in solvePermutationRight"
        exit(-1)
    return permX

def factorial(n):
    if( n == 1 ):
        return 1
    return n * factorial(n-1)

def reorderStatements(inputOutputMapping, dim):
    """ this function is used to minimize the amount of cache needed.
        It splits one statement into several for better locality.

        Input: {'Bi1i0i2': ['Ai1i0i2', 'Ai0i1i2'], 'Bi0i2i1': ['Ai0i2i1', 'Ai2i0i1'], 'Bi0i1i2': ['Ai0i1i2', 'Ai1i0i2'], 'Bi2i0i1': ['Ai2i0i1', 'Ai0i2i1'], 'Bi2i1i0': ['Ai2i1i0', 'Ai1i2i0'], 'Bi1i2i0': ['Ai1i2i0', 'Ai2i1i0']}
        Output: [{'Bi1i0i2': ['Ai1i0i2', 'Ai0i1i2'], 'Bi0i1i2': ['Ai0i1i2', 'Ai1i0i2']},
                 {'Bi0i2i1': ['Ai0i2i1', 'Ai2i0i1'], 'Bi2i0i1': ['Ai2i0i1', 'Ai0i2i1']}, 
                 {'Bi2i1i0': ['Ai2i1i0', 'Ai1i2i0'], 'Bi1i2i0': ['Ai1i2i0', 'Ai2i1i0']}]
    """

    #return [inputOutputMapping]

    connectivityMatrix = np.zeros((factorial(dim), factorial(dim)))
    inputOutputMappings = []

    # find connected components via breadth-first-search
    doneB = {}
    while( len(doneB) != len(inputOutputMapping) ):
        connectedB = []
        # find index which has not been processed yet
        for B in inputOutputMapping:
            if( not doneB.has_key(B) ):
                connectedB.append(B)
                break
        for B in connectedB:
            if( not doneB.has_key(B) ):
                for A in inputOutputMapping[B]:
                    # for every A find all other B's which also uses A
                    for B_ in inputOutputMapping:
                        if( B_ == B or hasItem(connectedB, B_) ):
                            continue
                        found = 0
                        for A_ in inputOutputMapping[B_]:
                            if( A_ == A):
                                found = 1
                                break
                        if( found ):
                            connectedB.append(B_)
                doneB[B] = 1
        localizedMapping = {}
        for B in connectedB:
            localizedMapping[B] = inputOutputMapping[B]
        inputOutputMappings.append(localizedMapping)
    #if( len(inputOutputMappings) > 1 ):
    #    print "Reorder optimization has been applied!"
    return inputOutputMappings

