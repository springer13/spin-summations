from utils import *
import itertools
import copy
import math

def getVectorizedKernels():
    ret = "static void store_tile(double *A, int lda, __m256d rows[4])\n"
    ret += "{\n"
    ret += "   _mm256_storeu_pd((A + 0 * lda), rows[0]);\n"
    ret += "   _mm256_storeu_pd((A + 1 * lda), rows[1]);\n"
    ret += "   _mm256_storeu_pd((A + 2 * lda), rows[2]);\n"
    ret += "   _mm256_storeu_pd((A + 3 * lda), rows[3]);\n"
    ret += "}\n"
    ret += "static void streaming_store_tile(double*A, int lda, __m256d rows[4])\n"
    ret += "{\n"
    ret += "   _mm256_stream_pd((A + 0 * lda), rows[0]);\n"
    ret += "   _mm256_stream_pd((A + 1 * lda), rows[1]);\n"
    ret += "   _mm256_stream_pd((A + 2 * lda), rows[2]);\n"
    ret += "   _mm256_stream_pd((A + 3 * lda), rows[3]);\n"
    ret += "}\n"
    ret += "static void load_tile(const double *A, int lda, __m256d rows[4])\n"
    ret += "{\n"
    ret += "   rows[0] = _mm256_loadu_pd((A + 0*lda));\n"
    ret += "   rows[1] = _mm256_loadu_pd((A + 1*lda));\n"
    ret += "   rows[2] = _mm256_loadu_pd((A + 2*lda));\n"
    ret += "   rows[3] = _mm256_loadu_pd((A + 3*lda));\n"
    ret += "}\n"
    ret += "\n"
    ret += "static void transpose_tile(__m256d rows[4])\n"
    ret += "{\n"
    ret += "   __m256d r4, r34, r3, r33;\n"
    ret += "   r33 = _mm256_shuffle_pd( rows[2], rows[3], 0x3 );\n"
    ret += "   r3 = _mm256_shuffle_pd( rows[0], rows[1], 0x3 );\n"
    ret += "   r34 = _mm256_shuffle_pd( rows[2], rows[3], 0xc );\n"
    ret += "   r4 = _mm256_shuffle_pd( rows[0], rows[1], 0xc );\n"
    ret += "   rows[0] = _mm256_permute2f128_pd( r34, r4, 0x2 );\n"
    ret += "   rows[1] = _mm256_permute2f128_pd( r33, r3, 0x2 );\n"
    ret += "   rows[2] = _mm256_permute2f128_pd( r33, r3, 0x13 );\n"
    ret += "   rows[3] = _mm256_permute2f128_pd( r34, r4, 0x13 );\n"
    ret += "}\n"
    ret += "static void scale_tile(__m256d rowsA[4], __m256d alpha)\n"
    ret += "{\n"
    ret += "   //FMA\n"
    ret += "   rowsA[0] = _mm256_mul_pd(alpha, rowsA[0]);\n"
    ret += "   rowsA[1] = _mm256_mul_pd(alpha, rowsA[1]);\n"
    ret += "   rowsA[2] = _mm256_mul_pd(alpha, rowsA[2]);\n"
    ret += "   rowsA[3] = _mm256_mul_pd(alpha, rowsA[3]);\n"
    ret += "}\n"
    ret += "//static void update_and_scale_tile(const __m256d rowsA[4], __m256d rowsB[4], __m256d alpha)\n"
    ret += "//{\n"
    ret += "//   rowsB[0] = _mm256_fmadd_pd(alpha, rowsA[0], rowsB[0]);\n"
    ret += "//   rowsB[1] = _mm256_fmadd_pd(alpha, rowsA[1], rowsB[1]);\n"
    ret += "//   rowsB[2] = _mm256_fmadd_pd(alpha, rowsA[2], rowsB[2]);\n"
    ret += "//   rowsB[3] = _mm256_fmadd_pd(alpha, rowsA[3], rowsB[3]);\n"
    ret += "//}\n"
    ret += "static void add_tile(const __m256d rowsA[4], __m256d rowsB[4])\n"
    ret += "{\n"
    ret += "   rowsB[0] = _mm256_add_pd(rowsA[0], rowsB[0]);\n"
    ret += "   rowsB[1] = _mm256_add_pd(rowsA[1], rowsB[1]);\n"
    ret += "   rowsB[2] = _mm256_add_pd(rowsA[2], rowsB[2]);\n"
    ret += "   rowsB[3] = _mm256_add_pd(rowsA[3], rowsB[3]);\n"
    ret += "}\n"
    ret += "static void sub_tile(const __m256d rowsA[4], __m256d rowsB[4])\n"
    ret += "{\n"
    ret += "   rowsB[0] = _mm256_sub_pd(rowsA[0], rowsB[0]);\n"
    ret += "   rowsB[1] = _mm256_sub_pd(rowsA[1], rowsB[1]);\n"
    ret += "   rowsB[2] = _mm256_sub_pd(rowsA[2], rowsB[2]);\n"
    ret += "   rowsB[3] = _mm256_sub_pd(rowsA[3], rowsB[3]);\n"
    ret += "}\n"

    if( 0 ):
        ret += "static void store_tile_float(float *A, int lda, __m256 rows[8])\n"
        ret += "{\n"
        ret += "   _mm256_store_ps((A + 0 * lda), rows[0]);\n"
        ret += "   _mm256_store_ps((A + 1 * lda), rows[1]);\n"
        ret += "   _mm256_store_ps((A + 2 * lda), rows[2]);\n"
        ret += "   _mm256_store_ps((A + 3 * lda), rows[3]);\n"
        ret += "   _mm256_store_ps((A + 4 * lda), rows[4]);\n"
        ret += "   _mm256_store_ps((A + 5 * lda), rows[5]);\n"
        ret += "   _mm256_store_ps((A + 6 * lda), rows[6]);\n"
        ret += "   _mm256_store_ps((A + 7 * lda), rows[7]);\n"
        ret += "}\n"
        ret += "static void streaming_store_tile_float(float *A, int lda, __m256 rows[8])\n"
        ret += "{\n"
        ret += "   _mm256_stream_ps((A + 0 * lda), rows[0]);\n"
        ret += "   _mm256_stream_ps((A + 1 * lda), rows[1]);\n"
        ret += "   _mm256_stream_ps((A + 2 * lda), rows[2]);\n"
        ret += "   _mm256_stream_ps((A + 3 * lda), rows[3]);\n"
        ret += "   _mm256_stream_ps((A + 4 * lda), rows[4]);\n"
        ret += "   _mm256_stream_ps((A + 5 * lda), rows[5]);\n"
        ret += "   _mm256_stream_ps((A + 6 * lda), rows[6]);\n"
        ret += "   _mm256_stream_ps((A + 7 * lda), rows[7]);\n"
        ret += "}\n"
        ret += "static void load_tile_float(const float *A, int lda, __m256 rows[8])\n"
        ret += "{\n"
        ret += "   rows[0] = _mm256_load_ps((A + 0*lda));\n"
        ret += "   rows[1] = _mm256_load_ps((A + 1*lda));\n"
        ret += "   rows[2] = _mm256_load_ps((A + 2*lda));\n"
        ret += "   rows[3] = _mm256_load_ps((A + 3*lda));\n"
        ret += "   rows[4] = _mm256_load_ps((A + 4*lda));\n"
        ret += "   rows[5] = _mm256_load_ps((A + 5*lda));\n"
        ret += "   rows[6] = _mm256_load_ps((A + 6*lda));\n"
        ret += "   rows[7] = _mm256_load_ps((A + 7*lda));\n"
        ret += "}\n"
        ret += "\n"
        ret += "static void transpose_tile_float(__m256 rows[8])\n"
        ret += "{\n"
        ret += "   __m256 r121, r139, r120, r138, r71, r89, r70, r88, r11, r1, r55, r29, r10, r0, r54, r28;\n"
        ret += "   r28 = _mm256_unpacklo_ps( rows[4], rows[5] );\n"
        ret += "   r54 = _mm256_unpacklo_ps( rows[6], rows[7] );\n"
        ret += "    r0 = _mm256_unpacklo_ps( rows[0], rows[1] );\n"
        ret += "   r10 = _mm256_unpacklo_ps( rows[2], rows[3] );\n"
        ret += "   r29 = _mm256_unpackhi_ps( rows[4], rows[5] );\n"
        ret += "   r55 = _mm256_unpackhi_ps( rows[6], rows[7] );\n"
        ret += "    r1 = _mm256_unpackhi_ps( rows[0], rows[1] );\n"
        ret += "   r11 = _mm256_unpackhi_ps( rows[2], rows[3] );\n"
        ret += "   r88 = _mm256_shuffle_ps( r28, r54, 0x44 );\n"
        ret += "   r70 = _mm256_shuffle_ps( r0, r10, 0x44 );\n"
        ret += "   r89 = _mm256_shuffle_ps( r28, r54, 0xee );\n"
        ret += "   r71 = _mm256_shuffle_ps( r0, r10, 0xee );\n"
        ret += "   r138 = _mm256_shuffle_ps( r29, r55, 0x44 );\n"
        ret += "   r120 = _mm256_shuffle_ps( r1, r11, 0x44 );\n"
        ret += "   r139 = _mm256_shuffle_ps( r29, r55, 0xee );\n"
        ret += "   r121 = _mm256_shuffle_ps( r1, r11, 0xee );\n"
        ret += "   rows[0] = _mm256_permute2f128_ps( r88, r70, 0x2 );\n"
        ret += "   rows[1] = _mm256_permute2f128_ps( r89, r71, 0x2 );\n"
        ret += "   rows[2] = _mm256_permute2f128_ps( r138, r120, 0x2 );\n"
        ret += "   rows[3] = _mm256_permute2f128_ps( r139, r121, 0x2 );\n"
        ret += "   rows[4] = _mm256_permute2f128_ps( r88, r70, 0x13 );\n"
        ret += "   rows[5] = _mm256_permute2f128_ps( r89, r71, 0x13 );\n"
        ret += "   rows[6] = _mm256_permute2f128_ps( r138, r120, 0x13 );\n"
        ret += "   rows[7] = _mm256_permute2f128_ps( r139, r121, 0x13 );\n"
        ret += "}\n"
        ret += "static void scale_tile_float(__m256 rowsA[8], __m256 alpha)\n"
        ret += "{\n"
        ret += "   //FMA\n"
        ret += "   rowsA[0] = _mm256_mul_ps(alpha, rowsA[0]);\n"
        ret += "   rowsA[1] = _mm256_mul_ps(alpha, rowsA[1]);\n"
        ret += "   rowsA[2] = _mm256_mul_ps(alpha, rowsA[2]);\n"
        ret += "   rowsA[3] = _mm256_mul_ps(alpha, rowsA[3]);\n"
        ret += "   rowsA[4] = _mm256_mul_ps(alpha, rowsA[4]);\n"
        ret += "   rowsA[5] = _mm256_mul_ps(alpha, rowsA[5]);\n"
        ret += "   rowsA[6] = _mm256_mul_ps(alpha, rowsA[6]);\n"
        ret += "   rowsA[7] = _mm256_mul_ps(alpha, rowsA[7]);\n"
        ret += "}\n"
        ret += "//static void update_and_scale_tile_float(const __m256 rowsA[8], __m256 rowsB[8], __m256 alpha)\n"
        ret += "//{\n"
        ret += "//   rowsB[0] = _mm256_fmadd_ps(alpha, rowsA[0], rowsB[0]);\n"
        ret += "//   rowsB[1] = _mm256_fmadd_ps(alpha, rowsA[1], rowsB[1]);\n"
        ret += "//   rowsB[2] = _mm256_fmadd_ps(alpha, rowsA[2], rowsB[2]);\n"
        ret += "//   rowsB[3] = _mm256_fmadd_ps(alpha, rowsA[3], rowsB[3]);\n"
        ret += "//   rowsB[4] = _mm256_fmadd_ps(alpha, rowsA[4], rowsB[4]);\n"
        ret += "//   rowsB[5] = _mm256_fmadd_ps(alpha, rowsA[5], rowsB[5]);\n"
        ret += "//   rowsB[6] = _mm256_fmadd_ps(alpha, rowsA[6], rowsB[6]);\n"
        ret += "//   rowsB[7] = _mm256_fmadd_ps(alpha, rowsA[7], rowsB[7]);\n"
        ret += "//}\n"
        ret += "static void add_tile_float(const __m256 rowsA[8], __m256 rowsB[8])\n"
        ret += "{\n"
        ret += "   rowsB[0] = _mm256_add_ps(rowsA[0], rowsB[0]);\n"
        ret += "   rowsB[1] = _mm256_add_ps(rowsA[1], rowsB[1]);\n"
        ret += "   rowsB[2] = _mm256_add_ps(rowsA[2], rowsB[2]);\n"
        ret += "   rowsB[3] = _mm256_add_ps(rowsA[3], rowsB[3]);\n"
        ret += "   rowsB[4] = _mm256_add_ps(rowsA[4], rowsB[4]);\n"
        ret += "   rowsB[5] = _mm256_add_ps(rowsA[5], rowsB[5]);\n"
        ret += "   rowsB[6] = _mm256_add_ps(rowsA[6], rowsB[6]);\n"
        ret += "   rowsB[7] = _mm256_add_ps(rowsA[7], rowsB[7]);\n"
        ret += "}\n"

    return ret

def getStride1Indices(statement):
    stride1indices = [0] # due to B
    for (scalar, perm) in statement:
        stride1indices.append(perm[0])
    stride1indices = list(set(stride1indices))
    return stride1indices 

def getLoopOrder(statement):
    dim = len(statement[0][1])
    loopOrder = []
    stride1Indices = getStride1Indices(statement)
    stride1Indices.sort(reverse=True) # make sure that 0 is the inner-most index
    fixedIndices = []
    for loopIdx in range(dim):
        fixed = 1
        for (scalar, perm) in statement:
            if( perm[loopIdx] != loopIdx ):
                fixed = 0
                break
        if ( fixed ):
            fixedIndices.append(loopIdx) # find all indices which are invariant across all permutations
    
    fixedIndices = list(set(fixedIndices) - set(stride1Indices))
    remainderIndices = list(set(range(dim)) - set(fixedIndices) - set(stride1Indices))
    loopOrder = fixedIndices + remainderIndices + stride1Indices

    if ( len(loopOrder) != dim ):
        print "ERROR: loopOrder dim not correct.\n"
        exit(-1)
    return loopOrder

def getMicroKernelCall(outputTensor, statement, idxB, inputOutputMapping, statement_id, loop_id, stride1Indices):

    if ( len(stride1Indices) > 2 ):
        print "ERROR: we rely on the fact that we have exactly two stride1 indices"
        exit(-1)
    code = "microKernel%d_loop%d<streamingStores>("%(statement_id, loop_id)
    for input_id in range(len(statement)):
        (scalar, permA) = statement[input_id]
        if ( permA[0] == stride1Indices[0] ):
            nonStride1Index = stride1Indices[1]
        else:
            nonStride1Index = stride1Indices[0]
        code += "&%s[IDXA(%s)], lda%d, "%(inputOutputMapping[outputTensor][input_id],arrayToString(applyPerm(permA,idxB)),permA.index(nonStride1Index)-1)
    
    code += "&%s[IDXB(%s)], ldb%d);\n"%(outputTensor,arrayToString(idxB), stride1Indices[1]-1)
    return code

def transposeRequired(stride1Indices, perm):
    stride1perm = []
    for idx in perm:
        if( idx != stride1Indices[0] and idx != stride1Indices[1] ):
            continue # skip non-stride1 indices
        stride1perm.append(idx)
    transposeRequired = 0
    if ( stride1perm[0] != stride1Indices[0] ): # we have to transpose the face if the indices are in the wrong order
        transposeRequired = 1
    return transposeRequired 

def genPrefetch(statement, statement_id, loop_id, blocking):
    code = "template<int streamingStores>\n"
    code += "void prefetch%d_loop%d("%(statement_id, loop_id)
    body = "   for(int i = 0; i < %d; i++){\n"%blocking
    for input_id in range(len(statement)):
        code += "const double *A%d, long lda%d, "%(input_id,input_id)
        body += "      _mm_prefetch((char*)(A%d + i * lda%d), _MM_HINT_T2);\n"%(input_id,input_id)

    body += "      if( !streamingStores )\n"
    body += "         _mm_prefetch((char*)(B + i * ldb), _MM_HINT_T2);\n"
    body += "   }\n"
    code = code + "double *B, long ldb){\n" + body 
    return code + "}\n"


def genMicroKernel(statement, statement_id, loop_id):
    stride1Indices = getStride1Indices(statement)
    if ( len(stride1Indices) > 2 ):
        print "ERROR: we rely on the fact that we have exactly two stride1 indices"
        exit(-1)

    scalarCode = "         B[0] = "
    body  = "   __m256d rows0[4];\n"
    body += "   __m256d rows1[4];\n"

    code = "template<int streamingStores>\n"
    code += "void microKernel%d_loop%d("%(statement_id, loop_id)
    for input_id in range(len(statement)):
        if ( input_id  == 0 ):
            workArray = 0
        else:
            workArray = 1
        code += "const double *A%d, long lda%d, "%(input_id,input_id)

        (scalar, permA) = statement[input_id]
        body += "   load_tile(A%d, lda%d, rows%d);\n"%(input_id,input_id, workArray)
        if ( len(stride1Indices) == 2 and transposeRequired(stride1Indices, permA) ):
            body += "   transpose_tile(rows%d);\n"%(workArray)
        scalarCode += "%.2f * A%d[0] + "%(scalar, input_id)
        body += "   __m256d alpha%d = _mm256_set1_pd(%f);\n"%(input_id,scalar)
        body += "   scale_tile(rows%d, alpha%d);\n"%(workArray, input_id)
        if ( input_id != 0 ):
            body += "   add_tile(rows%d, rows%d);\n\n"%(workArray, (workArray+1)%2)
        else:
            body += "\n"
        

    scalarCode = scalarCode[0:-3]+";\n"
    body += "   if( streamingStores == 1)\n"
    body += "      streaming_store_tile(B, ldb, rows0);\n"
    body += "   else\n"
    body += "      store_tile(B, ldb, rows0);\n"

    code = code + "double *B, long ldb){\n" + body 
    return code + "}\n"

def getPrefetchCall(microKernelCall, statement, prefetchDistance):
    """this is quite a hack"""
    prefetchCall = microKernelCall.replace("microKernel","prefetch")
    loopOrder = getLoopOrder(statement)
    # the stride-1 indices are not yet defined, we will set them to zero
    prefetchCall = prefetchCall.replace("i%d,"%loopOrder[-1],"0,")
    prefetchCall = prefetchCall.replace("i%d)"%loopOrder[-1],"0)")
    prefetchCall = prefetchCall.replace("i%d,"%loopOrder[-2],"0,")
    prefetchCall = prefetchCall.replace("i%d)"%loopOrder[-2],"0)")
    # prefetch along the direction of the inner-most loop which does not belong to a stride-1 index
    prefetchCall = prefetchCall.replace("i%d,"%loopOrder[-3],"(i%d+%d),"%(loopOrder[-3], prefetchDistance))
    prefetchCall = prefetchCall.replace("i%d)"%loopOrder[-3],"(i%d+%d))"%(loopOrder[-3], prefetchDistance))
    return prefetchCall
    
def getUpperBounds(tensorName, idxB):
    """ tensorName = Bi1i0i2
        idxB = (i0,i1,i2)
        return : map[i0] = blocking1, map[i1] = blocking0, map[i2] = blocking2
    """
    # splits "Bi1i0i2" int ["1","0","2"]
    tokens = [ tensorName[i+1:i+2] for i in range(1,len(tensorName),2)]
    upperBounds = {}
    for i in range(len(idxB)):
        upperBounds[idxB[i]] = "blocking" + tokens[i]

    return upperBounds


def genMacroKernel(inputOutputMapping, statement, indent, statement_id, loop_id, usePrefetching, useScalarVersion, disableCacheOpt  ):


    code = "{\n"
    indentLevel = 1
    loopOrder = getLoopOrder(statement)
    dim = len(statement[0][1])
    stride1Indices = getStride1Indices(statement)

    if( disableCacheOpt ):
        inputOutputMappings = [inputOutputMapping]
    else:
        inputOutputMappings = reorderStatements(inputOutputMapping, dim)

    for inputOutputMapping_ in inputOutputMappings:
        idxB = []
        for d in range(dim):
            idxB.append("i%d"%d)

        if ( useScalarVersion ):
            for d in loopOrder:
                increment = 1
                code += "%sfor(int i%d = 0; i%d < blocking; i%d += %d){\n"%(indent * indentLevel, d,d,d,increment)
                indentLevel += 1

            for outputTensor in inputOutputMapping_:
                line = "    %s%s[IDXB(%s)] = "%(indent * indentLevel, outputTensor, arrayToString(idxB))
                for input_id in range(len(statement)):
                    (scalar, permA) = statement[input_id]
                    line += "%.2f * %s[IDXA(%s)] + "%(scalar, inputOutputMapping_[outputTensor][input_id],arrayToString(applyPerm(permA,idxB)))
                code += line[0:-3] + ";\n"
            for d in loopOrder:
                indentLevel -= 1
                code += "%s}\n"%(indent * indentLevel)
        else:
            for d in loopOrder[0:-2]:
                increment = 1
                if ( hasItem(stride1Indices, d) and len(stride1Indices) == 2 ):
                    print "ERROR: non of these indices may be a stride-1 index, this is handled below"
                    exit(-1)
                code += "%sfor(int i%d = 0; i%d < blocking; i%d += %d){\n"%(indent * indentLevel, d,d,d,increment)
                indentLevel += 1

            for outputTensor in inputOutputMapping_:
                upperBounds = getUpperBounds(outputTensor, idxB)

                guardStr = ""
                for i in range(len(loopOrder)-2):
                    loopIdx = "i%d"%loopOrder[i]
                    guardStr += "%s < %s"%(loopIdx, upperBounds[loopIdx])
                    if ( i != len(loopOrder)-3 ):
                        guardStr += " && "
                body = "%sif ( %s )\n%s{\n"%(indent*indentLevel, guardStr,indent*indentLevel) 
                indentLevel += 1

                # vectorized Code
                if ( len(stride1Indices) <= 2 ):
                    microKernelCall = getMicroKernelCall(outputTensor, statement, idxB, inputOutputMapping_, statement_id, loop_id, [loopOrder[-1], loopOrder[-2]])
                    if( usePrefetching ):
                       prefetchDistance = 1 # TODO distinguish between 3D and 4D
                       prefetchCall = getPrefetchCall(microKernelCall, statement, prefetchDistance)
                       body += "%s%s"%(indent * indentLevel, prefetchCall)
                    incrementVec = 4
                    loopIdx0 = "i%d"%loopOrder[-2]
                    loopIdx1 = "i%d"%loopOrder[-1]
                    body += "%sfor(int %s = 0; %s < %s - (%d-1); %s += %d)\n"%(indent * indentLevel, loopIdx0, loopIdx0, upperBounds[loopIdx0], incrementVec,loopIdx0,incrementVec)
                    body += "%sfor(int %s = 0; %s < %s - (%d-1); %s += %d)\n"%(indent + indent * indentLevel, loopIdx1, loopIdx1, upperBounds[loopIdx1], incrementVec,loopIdx1,incrementVec)
                    body += "      %s%s"%(indent * indentLevel, microKernelCall)

                    # scalar remainder code
                    increment = 1
                    loopIdx0 = "i%d"%loopOrder[-2]
                    loopIdx1 = "i%d"%loopOrder[-1]
                    body += "%sfor(int %s = 0; %s < %s; %s += %d)\n"%(indent * indentLevel, loopIdx0, loopIdx0, upperBounds[loopIdx0], loopIdx0,increment)
                    body += "%sfor(int %s = 0; %s < %s; %s += %d)\n"%(indent + indent * indentLevel, loopIdx1, loopIdx1, upperBounds[loopIdx1], loopIdx1,increment)
                    body += "%sif( (%s >= ((%s/%d)*%d)) || (%s >= ((%s/%d)*%d)) )\n"%(indent * (indentLevel+2), loopIdx0, upperBounds[loopIdx0], incrementVec, incrementVec, loopIdx1, upperBounds[loopIdx1], incrementVec,incrementVec) 
                    indentLevel += 1
                    line = "   %s%s[IDXB(%s)] = "%(indent * (indentLevel+1), outputTensor, arrayToString(idxB))
                    for input_id in range(len(statement)):
                        (scalar, permA) = statement[input_id]
                        line += "%.2f * %s[IDXA(%s)] + "%(scalar, inputOutputMapping_[outputTensor][input_id],arrayToString(applyPerm(permA,idxB)))
                    body += line[0:-3].replace("+ -","-") + ";\n"
                    indentLevel -= 1
                    body += "%s}\n"%(indent * indentLevel)
                    code += body
                    if ( len(stride1Indices) == 2 ):
                        indentLevel -= 1

            
            for d in loopOrder[0:-2]:
                indentLevel -= 1
                code += "%s}\n"%(indent * indentLevel)
    code += "}\n"
    return code

def getMaxBlocks(dim):
    """ return maximum number of auxiliary BxBxB... blocks """
    loopIndices = []
    maxNumBlocks = 0
    for d in range(1,dim+1):
        loopIndices.append("i%d"%d)
        indicesB = createAllTupple(loopIndices, dim)
        maxNumBlocks = max(maxNumBlocks, len(indicesB))
    return maxNumBlocks 

def genHPC_wrapper(statements, h_fp, cpp_fp, blocking, useScalarVersion,
        usePrefetching, useStreamingStores, _remainder, disableCacheOpt,
        functionName ):
    if( not (blocking == 8 or  blocking == 4 or blocking == 16) ):
        print "ERROR: blocking_thread must be either 4,8"
        exit(-1)
    dim = len(statements[0][0][1])
    ldaStr = ""
    for i in range(dim-1):
        ldaStr += "long lda%d, "%i
    ldaStr = ldaStr[0:-2]
    ldaStrCall = ldaStr.replace("long ","")
    ldbStr = ldaStr.replace('a','b')
    ldbStrCall = ldbStr.replace("long ","")
    ldwStr = ldaStr.replace('a','w')
    ldwStrCall = ldwStr.replace("long ","")

    genHPC_implementation(statements, h_fp, blocking, useScalarVersion,
            usePrefetching, disableCacheOpt, functionName )

    header = "size_t %s(const double* __restrict__ A, %s,\n   double* __restrict__ B, %s, const int n, double * __restrict__ work_, const int numThreads)"%(functionName, ldaStr, ldbStr)
    h_fp.write(header + ";\n")
    code = header + "\n{\n"
    maxNumBlocks = getMaxBlocks(dim)
    numTempBlocks = len(statements)-1
    blockSize = blocking**dim
    code += "   if( A == NULL ) return %d * %d * %d * numThreads * sizeof(double);\n"%(blockSize, maxNumBlocks, numTempBlocks)
    if ( useStreamingStores ):
        code += "   int streamingStores = ( (((size_t)B)%32) == 0 && ((ldb0 * sizeof(double)) % 32) == 0 ) ? 1 : 0;\n"
    else:
        code += "   int streamingStores = 0;\n"
    code += "#pragma omp parallel num_threads(numThreads)\n"
    code += "{\n"
    code += "#pragma omp single\n"
    remainder = 1
    code += "   if ( streamingStores == 1){\n"
    elseStr = ""
    filename = "transpose_intern.h"
    fp = open(filename, "w")
    for streamingStores in [0,1]:
        for remainder in reversed(range(0,blocking)):
            if( _remainder != -1 and remainder != _remainder ): #for faster compilation
                continue
            # split templated calles into different files for faster compilation
            header_intern = "void %s_intern_%d_%d_%d(const double* __restrict__ A, %s,\n   double* __restrict__ B, %s, const int n, double * __restrict__ work_)"%(functionName, streamingStores, blocking, remainder, ldaStr, ldbStr)
            fp.write(header_intern + ";\n")
            codeIntern = "#include \"transpose.h\"\n"
            codeIntern += header_intern + "\n{\n"
            codeIntern += "   %s_intern<%d,%d, %d>(A, %s, B, %s, n, work_);\n"%(functionName, streamingStores, blocking, remainder, ldaStrCall, ldbStrCall)
            codeIntern += "}\n"
            filename = "transpose%d_%d.cpp"%(streamingStores, remainder)
            fp_cpp_intern = open(filename, "w")
            fp_cpp_intern.write(codeIntern)
            fp_cpp_intern.close()
    fp.close()

    for remainder in reversed(range(0,blocking)):
        if( _remainder != -1 and remainder != _remainder ): #for faster compilation
            continue
        code += "      %sif ( n %% %d == %d )\n"%(elseStr, blocking, remainder)
        code += "         %s_intern_1_%d_%d(A, %s, B, %s, n, work_);\n"%(functionName, blocking, remainder, ldaStrCall, ldbStrCall)
        elseStr = "else "
    code += "   }else{\n"
    elseStr = ""
    for remainder in reversed(range(0,blocking)):
        if( _remainder != -1 and remainder != _remainder ): #for faster compilation
            continue
        code += "      %sif ( n %% %d == %d )\n"%(elseStr, blocking, remainder)
        code += "         %s_intern_0_%d_%d(A, %s, B, %s, n, work_);\n"%(functionName, blocking, remainder, ldaStrCall, ldbStrCall)
        elseStr = "else "
    code += "   }\n"
    code += "} // omp parallel\n"
    code += "   return 0;\n"
    code += "}\n"
    cpp_fp.write(code)


def genHPC_implementation(statements, cpp_fp, blockingSize, useScalarVersion,
        usePrefetching, disableCacheOpt, functionName  ):
    if ( not useScalarVersion ):
        cpp_fp.write(getVectorizedKernels())

    dim = len(statements[0][0][1])
    ldaStr = ""
    for i in range(dim-1):
        ldaStr += "long lda%d, "%i
    ldaStr = ldaStr[0:-2]
    ldaStrCall = ldaStr.replace("long ","")
    ldbStr = ldaStr.replace('a','b')
    ldbStrCall = ldbStr.replace("long ","")
    ldwStr = ldaStr.replace('a','w')
    ldwStrCall = ldwStr.replace("long ","")

    code = ""
    loopIndices = []
    macroKernelTaskOld = ""
    macroKernelsCode  = ""
    blockingsStr = " int blocking, int blockingRemainder"
    for loopLevel in reversed(range(-1,dim)):

        # generate header
        header = "template<int streamingStores, %s>\n"%blockingsStr
        header += "void %s_intern(const double* __restrict__ A, %s,\n   double* __restrict__ B, %s, const int num, double * __restrict__ work_"%(functionName, ldaStr, ldbStr)
        for loopIndex in loopIndices:
            header += ", const int %s"%loopIndex
        header += ")"
        functionCode = header + "\n{\n"
        maxNumBlocks = getMaxBlocks(dim)
        numTempBlocks = len(statements)-1

        functionCode += "   const int ldw0 = blocking;\n"
        functionCode += "   const int ldw1 = BLOCKING2;\n"
        functionCode += "   const int ldw2 = BLOCKING3;\n"

        # generate loops
        indentLevel = 1
        indent = "   "
        if ( loopLevel == dim-1 ):
            start = "0"
        else:
            start = "i%d + blocking"%(loopLevel+1)
        functionCode += macroKernelTaskOld
        if( loopLevel  >= 0):
            functionCode += "%sint i%d;\n"%(indent*indentLevel, loopLevel)
            functionCode += "%sfor(i%d = %s; i%d < num - (blocking - 1); i%d += blocking)\n"%(indent * indentLevel, loopLevel, start, loopLevel,loopLevel)
            loopIndices.append("i%d"%loopLevel)
        indentLevel += 1

        workIdMapping = {}
        declareVariables = {}
        macroKernelTask = ""
        for statement_id in range(len(statements)):
            statement = statements[statement_id]

            inputOutputMapping = {}
            inputTensorsCallToDef = {}
            inputTensors = []
            outputTensorsDef = ""
            outputTensorsCall = ""
            inputTensorsDef = ""
            inputTensorsCall = ""
            workIdMappingNew = {}

            # generate all permutations of the input indices
            indicesB = createAllTupple(loopIndices, dim)
            for idxB in indicesB: 
                if ( statement_id == len(statements)-1 ):
                    originalLocation = "&B[IDXB(%s)]"%arrayToString(idxB)
                    declareVariables[originalLocation] = "B%s_"%arrayToString(idxB,1)
                    outputTensorsCall += "%s, "%(declareVariables[originalLocation])
                else:
                    workVariable = arrayToString(idxB)
                    if( not workIdMappingNew.has_key(workVariable) ):
                        workIdMappingNew[workVariable] = len(workIdMappingNew)
                    originalLocation = "&work[IDXW_OUTER(%s,%d)]"%(workIdMappingNew[workVariable],statement_id%2)
                    declareVariables[originalLocation] = "work%s_%d_"%(workIdMappingNew[workVariable],statement_id%2) 
                    outputTensorsCall += "%s, "%(declareVariables[originalLocation])

                outputTensorDef = "B%s"%arrayToString(idxB,1)
                outputTensorsDef += "double *%s, "%(outputTensorDef)

                inputOutputMapping[outputTensorDef] = []
                # apply permutation of current statement to indput index
                for (scalar, permA) in statement: 
                    originalLocation = "&A[IDXA(%s)]"%(arrayToString(applyPerm(permA,idxB)))
                    declareVariables[originalLocation] = "A%s_"%(arrayToString(applyPerm(permA,idxB),1)) 
                    inputTensorCall = declareVariables[originalLocation]
                    if ( statement_id != 0 ):
                        workVariable = arrayToString(applyPerm(permA,idxB))
                        originalLocation = "&work[IDXW_OUTER(%s,%d)]"%(workIdMapping[workVariable],(statement_id-1)%2)
                        declareVariables[originalLocation] = "work%s_%d_"%(workIdMapping[workVariable],(statement_id-1)%2) 
                        inputTensorCall = declareVariables[originalLocation]

                    # lookup input
                    if( not inputTensorsCallToDef.has_key(inputTensorCall) ):
                        inputTensorDef = "A%s"%arrayToString(applyPerm(permA,idxB),1)
                        inputTensors.append(inputTensorDef)
                        inputTensorsCallToDef[inputTensorCall] = inputTensorDef
                        inputTensorsDef += "const double *%s, "%(inputTensorDef)
                        inputTensorsCall += "%s, "%inputTensorCall
                    

                    inputOutputMapping[outputTensorDef].append(inputTensorsCallToDef[inputTensorCall])

            workIdMapping = copy.deepcopy(workIdMappingNew)

            ldInCall = ldaStrCall
            ldOutCall = ldbStrCall
            blockingsStrNextCall = "%s,blocking%s"%(blockingsStr.replace("int ",""), loopIndices[-1].replace("i",""))
            blockingsStrNext = "%s,int blocking%s"%(blockingsStr, loopIndices[-1].replace("i",""))
            templateStr = "<streamingStores, %s>"%(blockingsStrNextCall)
            if ( inputTensorsCall.find("work") != -1 ):
                ldInCall = ldwStrCall
            if ( outputTensorsCall.find("work") != -1 ):
                ldOutCall = ldwStrCall
                templateStr = "<0,%s>"%(blockingsStrNextCall)
            if ( loopLevel >= 0):
                macroKernelTask += "%smacroKernel%d_loop%d%s(%s, %s,\n%s   %s, %s);\n"%(indentLevel * indent,statement_id, loopLevel, templateStr, inputTensorsCall[0:-2], ldInCall, indentLevel * indent, outputTensorsCall[0:-2], ldOutCall)
                macroKernelHeader = "template<int streamingStores,%s>\n"%blockingsStrNext
                macroKernelHeader += "void macroKernel%d_loop%d(%s, %s,\n   %s, %s)\n"%(statement_id, loopLevel, inputTensorsDef[0:-2], ldaStr, outputTensorsDef[0:-2], ldbStr)

            if ( loopLevel >= 0):
                usePrefetchingTmp = usePrefetching 
                if( usePrefetchingTmp ):
                    usePrefetchingTmp = statement_id == 0
                if( len(getStride1Indices(statement)) <= 2 ): # ==1 doesn't require a microKernel
                    if ( not useScalarVersion ):
                        macroKernelsCode += genMicroKernel(statement, statement_id, loopLevel)
                        macroKernelsCode += genPrefetch(statement, statement_id, loopLevel, blockingSize)
                macroKernelsCode += macroKernelHeader + genMacroKernel(inputOutputMapping, statement, indent, statement_id, loopLevel, usePrefetchingTmp, useScalarVersion, disableCacheOpt )
        declare = ""
        for inPtr in declareVariables:
            const = ""
            if ( inPtr.find("A") != -1 ):
                const = "const "
            declare += "%s%sdouble *%s = %s;\n"%(indent * indentLevel, const, declareVariables[inPtr], inPtr)
        declare = "%sdouble *work = &work_[BLOCKING%d * %d * %d * omp_get_thread_num()];\n"%(indent * indentLevel, dim, maxNumBlocks, numTempBlocks) + declare
        firstPrivate = "A, B, work_, num, %s, %s, %s"%(ldaStrCall, ldbStrCall, ldwStrCall )
        if( loopLevel >= 0):
            recursiveCall = "%s%s_intern<streamingStores,%s, blocking>(A, %s, B, %s, num, work_"%(indent * indentLevel,functionName, blockingsStr.replace("int ",""), ldaStrCall, ldbStrCall)
        for idx in loopIndices:
            firstPrivate += ", %s"%idx
            if( loopLevel >= 0):
                recursiveCall += ", %s"%idx
        if( loopLevel >= 0):
            recursiveCall += ");\n"
        macroKernelTask = "#pragma omp task firstprivate(%s)\n%s{\n"%(firstPrivate, indent * (indentLevel-1)) + declare + macroKernelTask + "%s}\n"%(indent * (indentLevel-1))
        macroKernelTaskOld = macroKernelTask 
        if( loopLevel >= 0):
            functionCode += recursiveCall 

        indentLevel -= 1
        if( loopLevel >= 0):
            functionCode += "%sif( blockingRemainder > 0 && (%s + blockingRemainder <= num) )\n"%(indent*indentLevel,loopIndices[-1])
            functionCode += "%s%s_intern<streamingStores,%s, blockingRemainder>(A, %s, B, %s, num, work_"%(indent +indent * indentLevel,functionName, blockingsStr.replace("int ",""), ldaStrCall, ldbStrCall)
            for idx in loopIndices:
                functionCode  += ", %s"%idx
            functionCode += ");\n"

        functionCode  += "}\n"
        code = functionCode + code
        blockingsStr  += ", int blocking%s"%loopIndices[-1].replace("i","")

    code = macroKernelsCode + code
    cpp_fp.write(code)




