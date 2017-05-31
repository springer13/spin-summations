import copy
import multiprocessing
import time
import subprocess
import sys
import shutil
import glob
import main
import os
import itertools
from utils import *

import hpc_implementation
import reference

OKGREEN = '\033[92m'
FAIL = '\033[91m'
WARNING = '\033[93m'
ENDC = '\033[0m'
DEVNULL = open(os.devnull, 'wb')

statements = [
        [[(2.0,[0,1,2]), (-1.0,[2,1,0]), (-1.0,[0,2,1])], 
         [(2.0,[0,1,2]), (-1.0,[1,0,2])]], # (2-P(ab)) (2-P(ac)-P(bc)) A[abc]
        [[(2.0,[0,1,2]), (-1.0,[2,1,0]), (-1.0,[0,2,1])]], # (2-P(ac)-P(bc)) A[abc]
        [[(2.0,[0,1,2]), (-1.0,[1,0,2]), (-1.0,[0,2,1])]], # (2-P(ab)-P(bc)) A[abc]
        [[(2.0,[0,1,2]), (-1.0,[1,0,2]), (-1.0,[2,1,0])]], # (2-P(ab)-P(ac)) A[abc]
        [[(2.0,[0,1,2,3]), (-1.0,[3,1,2,0]), (-1.0,[0,3,2,1]), (-1.0,[0,1,3,2])], # (2-P(ad)-P(bd)-P(cd)) 
         [(2.0,[0,1,2,3]), (-1.0,[2,1,0,3]), (-1.0,[0,2,1,3])],                   # (2-P(ac)-P(bc))
         [(2.0,[0,1,2,3]), (-1.0,[1,0,2,3])]],                                    # (2-P(ab)) 
        [[(2.0,[0,1,2,3]), (-1.0,[3,1,2,0]), (-1.0,[0,3,2,1]), (-1.0,[0,1,3,2])], # (2-P(ad)-P(bd)-P(cd)) 
         [(2.0,[0,1,2,3]), (-1.0,[2,1,0,3]), (-1.0,[0,2,1,3])]],                  # (2-P(ac)-P(bc)) 
        [[(2.0,[0,1,2,3]), (-1.0,[3,1,2,0]), (-1.0,[0,3,2,1]), (-1.0,[0,1,3,2])], # (2-P(ad)-P(bd)-P(cd)) 
         [(2.0,[0,1,2,3]), (-1.0,[1,0,2,3]), (-1.0,[0,2,1,3])]],                  # (2-P(ab)-P(bc)) 
        [[(2.0,[0,1,2,3]), (-1.0,[3,1,2,0]), (-1.0,[0,3,2,1]), (-1.0,[0,1,3,2])], # (2-P(ad)-P(bd)-P(cd)) 
         [(2.0,[0,1,2,3]), (-1.0,[1,0,2,3]), (-1.0,[2,1,0,3])]],                  # (2-P(ab)-P(ac)) 

        [[(2.0,[0,1,2,3]), (-1.0,[2,1,0,3]), (-1.0,[0,2,1,3]), (-1.0,[0,1,3,2])], # (2-P(ac)-P(bc)-P(cd))
         [(2.0,[0,1,2,3]), (-1.0,[3,1,2,0]), (-1.0,[0,3,2,1])]],                  # (2-P(ad)-P(bd)) 
        [[(2.0,[0,1,2,3]), (-1.0,[2,1,0,3]), (-1.0,[0,2,1,3]), (-1.0,[0,1,3,2])], # (2-P(ac)-P(bc)-P(cd))
         [(2.0,[0,1,2,3]), (-1.0,[1,0,2,3]), (-1.0,[0,3,2,1])]],                  # (2-P(ab)-P(bd)) 
        [[(2.0,[0,1,2,3]), (-1.0,[2,1,0,3]), (-1.0,[0,2,1,3]), (-1.0,[0,1,3,2])], # (2-P(ac)-P(bc)-P(cd))
         [(2.0,[0,1,2,3]), (-1.0,[1,0,2,3]), (-1.0,[3,1,2,0])]],                  # (2-P(ab)-P(ad)) 

        [[(2.0,[0,1,2,3]), (-1.0,[1,0,2,3]), (-1.0,[0,2,1,3]), (-1.0,[0,3,2,1])], # (2-P(ab)-P(bc)-P(bd))  
         [(2.0,[0,1,2,3]), (-1.0,[3,1,2,0]), (-1.0,[0,1,3,2])]],                  # (2-P(ad)-P(cd)) 
        [[(2.0,[0,1,2,3]), (-1.0,[1,0,2,3]), (-1.0,[0,2,1,3]), (-1.0,[0,3,2,1])], # (2-P(ab)-P(bc)-P(bd))  
         [(2.0,[0,1,2,3]), (-1.0,[2,1,0,3]), (-1.0,[0,1,3,2])]],                  # (2-P(ac)-P(cd)) 
        [[(2.0,[0,1,2,3]), (-1.0,[1,0,2,3]), (-1.0,[0,2,1,3]), (-1.0,[0,3,2,1])], # (2-P(ab)-P(bc)-P(bd))  
         [(2.0,[0,1,2,3]), (-1.0,[2,1,0,3]), (-1.0,[3,1,2,0])]],                  # (2-P(ac)-P(ad)) 

        [[(2.0,[0,1,2,3]), (-1.0,[1,0,2,3]), (-1.0,[2,1,0,3]), (-1.0,[3,1,2,0])], # (2-P(ab)-P(ac)-P(ad)) 
         [(2.0,[0,1,2,3]), (-1.0,[0,3,2,1]), (-1.0,[0,1,3,2])]],                  # (2-P(bd)-P(cd))  
        [[(2.0,[0,1,2,3]), (-1.0,[1,0,2,3]), (-1.0,[2,1,0,3]), (-1.0,[3,1,2,0])], # (2-P(ab)-P(ac)-P(ad)) 
         [(2.0,[0,1,2,3]), (-1.0,[0,2,1,3]), (-1.0,[0,1,3,2])]],                  # (2-P(bc)-P(cd))
        [[(2.0,[0,1,2,3]), (-1.0,[1,0,2,3]), (-1.0,[2,1,0,3]), (-1.0,[3,1,2,0])], # (2-P(ab)-P(ac)-P(ad)) 
         [(2.0,[0,1,2,3]), (-1.0,[0,2,1,3]), (-1.0,[0,3,2,1])]],                  # (2-P(bc)-P(bd))  

        [[(2.0,[0,1,2,3]), (-1.0,[3,1,2,0]), (-1.0,[0,3,2,1]), (-1.0,[0,1,3,2])]], # (2-P(ad)-P(bd)-P(cd)) 
        [[(2.0,[0,1,2,3]), (-1.0,[2,1,0,3]), (-1.0,[0,2,1,3]), (-1.0,[0,1,3,2])]], # (2-P(ac)-P(bc)-P(cd))
        [[(2.0,[0,1,2,3]), (-1.0,[1,0,2,3]), (-1.0,[0,2,1,3]), (-1.0,[0,3,2,1])]], # (2-P(ab)-P(bc)-P(bd))  
        [[(2.0,[0,1,2,3]), (-1.0,[1,0,2,3]), (-1.0,[2,1,0,3]), (-1.0,[3,1,2,0])]]  # (2-P(ab)-P(ac)-P(ad)) 
        ]

# most of these statements are similar to those above. However, some are modified s.t. each statement has at most two different stride-1 indices
statements_regularized = [
        [[(2.0,[0,1,2]), (-1.0,[2,1,0]), (-1.0,[0,2,1])], 
         [(2.0,[0,1,2]), (-1.0,[1,0,2])]], # (2-P(ab)) (2-P(ac)-P(bc)) A[abc]
        [[(2.0,[0,1,2]), (-1.0,[2,1,0]), (-1.0,[0,2,1])]], # (2-P(ac)-P(bc)) A[abc]
        [[(2.0,[0,1,2]), (-1.0,[1,0,2]), (-1.0,[0,2,1])]], # (2-P(ab)-P(bc)) A[abc]
        [[(1.0,[1,0,2])],
         [(2.0,[1,0,2]), (-1.0,[0,1,2]), (-1.0,[1,2,0])]], # (2-P(ab)-P(ac)) A[abc]
        [[(2.0,[0,1,2,3]), (-1.0,[3,1,2,0]), (-1.0,[0,3,2,1]), (-1.0,[0,1,3,2])], # (2-P(ad)-P(bd)-P(cd)) 
         [(2.0,[0,1,2,3]), (-1.0,[2,1,0,3]), (-1.0,[0,2,1,3])],                   # (2-P(ac)-P(bc))
         [(2.0,[0,1,2,3]), (-1.0,[1,0,2,3])]],                                    # (2-P(ab)) 
        [[(2.0,[0,1,2,3]), (-1.0,[3,1,2,0]), (-1.0,[0,3,2,1]), (-1.0,[0,1,3,2])], # (2-P(ad)-P(bd)-P(cd)) 
         [(2.0,[0,1,2,3]), (-1.0,[2,1,0,3]), (-1.0,[0,2,1,3])]],                  # (2-P(ac)-P(bc)) 
        [[(2.0,[0,1,2,3]), (-1.0,[3,1,2,0]), (-1.0,[0,3,2,1]), (-1.0,[0,1,3,2])], # (2-P(ad)-P(bd)-P(cd)) 
         [(2.0,[0,1,2,3]), (-1.0,[1,0,2,3]), (-1.0,[0,2,1,3])]],                  # (2-P(ab)-P(bc)) 
        [[(2.0,[0,1,2,3]), (-1.0,[3,1,2,0]), (-1.0,[0,3,2,1]), (-1.0,[0,1,3,2])], # (2-P(ad)-P(bd)-P(cd)) 
         [(1.0,[1,0,2,3])],                  # 7
         [(2.0,[1,0,2,3]), (-1.0,[0,1,2,3]), (-1.0,[1,2,0,3])]],                  # (2-P(ab)-P(ac)) 

        [[(2.0,[0,1,2,3]), (-1.0,[2,1,0,3]), (-1.0,[0,2,1,3]), (-1.0,[0,1,3,2])], # (2-P(ac)-P(bc)-P(cd))
         [(2.0,[0,1,2,3]), (-1.0,[3,1,2,0]), (-1.0,[0,3,2,1])]],                  # (2-P(ad)-P(bd)) 
        [[(2.0,[0,1,2,3]), (-1.0,[2,1,0,3]), (-1.0,[0,2,1,3]), (-1.0,[0,1,3,2])], # (2-P(ac)-P(bc)-P(cd))
         [(2.0,[0,1,2,3]), (-1.0,[1,0,2,3]), (-1.0,[0,3,2,1])]],                  # (2-P(ab)-P(bd)) 
        [[(2.0,[0,1,2,3]), (-1.0,[2,1,0,3]), (-1.0,[0,2,1,3]), (-1.0,[0,1,3,2])], # (2-P(ac)-P(bc)-P(cd))
         [(1.0,[1,0,2,3])],                  # 10
         [(2.0,[1,0,2,3]), (-1.0,[0,1,2,3]), (-1.0,[1,3,2,0])]],                  # (2-P(ab)-P(ad)) 

        [[(2.0,[0,1,2,3]), (-1.0,[1,0,2,3]), (-1.0,[0,2,1,3]), (-1.0,[0,3,2,1])], # (2-P(ab)-P(bc)-P(bd))  
         [(2.0,[0,1,2,3]), (-1.0,[3,1,2,0]), (-1.0,[0,1,3,2])]],                  # (2-P(ad)-P(cd)) 
        [[(2.0,[0,1,2,3]), (-1.0,[1,0,2,3]), (-1.0,[0,2,1,3]), (-1.0,[0,3,2,1])], # (2-P(ab)-P(bc)-P(bd))  
         [(2.0,[0,1,2,3]), (-1.0,[2,1,0,3]), (-1.0,[0,1,3,2])]],                  # (2-P(ac)-P(cd)) 
        [[(2.0,[0,1,2,3]), (-1.0,[1,0,2,3]), (-1.0,[0,2,1,3]), (-1.0,[0,3,2,1])], # (2-P(ab)-P(bc)-P(bd))  
         [(1.0,[1,0,2,3])],                  # 13
         [(2.0,[1,0,2,3]), (-1.0,[1,2,0,3]), (-1.0,[1,3,2,0])]],                  # (2-P(ac)-P(ad)) 

        [[(1.0,[1,0,2,3])],                  #
         [(2.0,[1,0,2,3]), (-1.0,[0,1,2,3]), (-1.0,[1,2,0,3]), (-1.0,[1,3,2,0])], # (2-P(ab)-P(ac)-P(ad)) 
         [(2.0,[0,1,2,3]), (-1.0,[0,3,2,1]), (-1.0,[0,1,3,2])]],                  # (2-P(bd)-P(cd))  
        [[(1.0,[1,0,2,3])],                  #
         [(2.0,[1,0,2,3]), (-1.0,[0,1,2,3]), (-1.0,[1,2,0,3]), (-1.0,[1,3,2,0])], # (2-P(ab)-P(ac)-P(ad)) 
         [(2.0,[0,1,2,3]), (-1.0,[0,2,1,3]), (-1.0,[0,1,3,2])]],                  # (2-P(bc)-P(cd))
        [[(1.0,[1,0,2,3])],                  #
         [(2.0,[1,0,2,3]), (-1.0,[0,1,2,3]), (-1.0,[1,2,0,3]), (-1.0,[1,3,2,0])], # (2-P(ab)-P(ac)-P(ad)) 
         [(2.0,[0,1,2,3]), (-1.0,[0,2,1,3]), (-1.0,[0,3,2,1])]],                  # (2-P(bc)-P(bd))  

        [[(2.0,[0,1,2,3]), (-1.0,[3,1,2,0]), (-1.0,[0,3,2,1]), (-1.0,[0,1,3,2])]], # (2-P(ad)-P(bd)-P(cd)) 
        [[(2.0,[0,1,2,3]), (-1.0,[2,1,0,3]), (-1.0,[0,2,1,3]), (-1.0,[0,1,3,2])]], # (2-P(ac)-P(bc)-P(cd))
        [[(2.0,[0,1,2,3]), (-1.0,[1,0,2,3]), (-1.0,[0,2,1,3]), (-1.0,[0,3,2,1])]], # (2-P(ab)-P(bc)-P(bd))  
        [[(1.0,[1,0,2,3])],                  #
         [(2.0,[1,0,2,3]), (-1.0,[0,1,2,3]), (-1.0,[1,2,0,3]), (-1.0,[1,3,2,0])]]] # (2-P(ab)-P(ac)-P(ad)) 

#        [[(1.0,[1,0,2,3])]], #
#        [[(1.0,[1,2,0,3])]], #
#        [[(1.0,[1,3,0,2])]]]

def convertToOldInput(statements):
    dim = len(statements[0][1])
    if ( dim == 4 ):
        idxB = ['a','b','c','d']
    else:
        idxB = ['a','b','c']

    line = "B[%s] = "%arrayToString(idxB)
    for (scalar,perm) in statements:
        line += "%d*A[%s] + "%(int(scalar), arrayToString(applyPerm(perm,idxB)))
    line = line.replace("+ -","-")
    return line[0:-3]

def removeOldFiles():
    for filename in glob.glob(r'./*.cpp'):
        os.remove(filename)
    for filename in glob.glob(r'./*.h'):
        os.remove(filename)


def benchmark(numThreads, generateOnly, useScalarVersion, usePrefetching,
        useStreamingStores, testcase, inplace, disableCacheOpt ):
    affinity = "compact,1"

    sizeRange = {}
    sizeRange[3] = [16*13, 16*16, 16*22, 16*28,16*24] #68MB, 128,332,686, 1228
    sizeRange[4] = [8*7,8*8, 8*10, 8*12, 8*14]        #75MB, 128,312,648, 1200
    if( len(sizeRange[3]) != len(sizeRange[4]) ):
        print "ERROR"
        return

    measurements_minFlops_flops = [[] for i in range(len(sizeRange[3]))]
    measurements_minFlopsTempAB_flops = [[] for i in range(len(sizeRange[3]))]
    measurements_minFlopsTempABFused_flops = [[] for i in range(len(sizeRange[3]))]
    measurements_hpc_flops = [[] for i in range(len(sizeRange[3]))]
    measurements_minFlopsReg_flops = [[] for i in range(len(sizeRange[3]))]
    measurements_tempAB_flops = [[] for i in range(len(sizeRange[3]))]
    measurements_spatialB_flops = [[] for i in range(len(sizeRange[3]))]
    measurements_minFlops_minMetric = [[] for i in range(len(sizeRange[3]))]
    measurements_minFlopsTempAB_minMetric = [[] for i in range(len(sizeRange[3]))]
    measurements_minFlopsTempABFused_minMetric = [[] for i in range(len(sizeRange[3]))]
    measurements_hpc_minMetric = [[] for i in range(len(sizeRange[3]))]
    measurements_minFlopsReg_minMetric = [[] for i in range(len(sizeRange[3]))]
    measurements_tempAB_minMetric = [[] for i in range(len(sizeRange[3]))]
    measurements_spatialB_minMetric = [[] for i in range(len(sizeRange[3]))]
    measurements_minFlops_maxMetric = [[] for i in range(len(sizeRange[3]))]
    measurements_minFlopsTempAB_maxMetric = [[] for i in range(len(sizeRange[3]))]
    measurements_minFlopsTempABFused_maxMetric = [[] for i in range(len(sizeRange[3]))]
    measurements_hpc_maxMetric = [[] for i in range(len(sizeRange[3]))]
    measurements_minFlopsReg_maxMetric = [[] for i in range(len(sizeRange[3]))]
    measurements_tempAB_maxMetric = [[] for i in range(len(sizeRange[3]))]
    measurements_spatialB_maxMetric = [[] for i in range(len(sizeRange[3]))]
    maxBW_minMetric = -1
    maxBW_maxMetric = -1
    for statement_id in range(len(statements)):
        statement = statements[statement_id]
        statement_reg = statements_regularized[statement_id]
        dim = len(statement[0][0][1])
        if ( inplace and len(statement_reg) == 1): # fix for inplcae if the current statement if it doesn't have multiple statements
            statement_reg = [[((1.0),range(dim))],statement_reg[0]]
        if ( testcase != -1 and statement_id != testcase ):
            continue #skip all non-selected testcases

        for size_id in range(len(sizeRange[dim])):
            size = sizeRange[dim][size_id]
            if ( size_id != 2):
                continue
            blocking = 16
            if ( dim == 4 ):
                blocking = 8
            size = ((size+blocking-1)/blocking)*blocking +0 # TODO remove: round to next multiple of blocking
            print "tensor size: %.2f MiB\n"%(size**dim*8/1024./1024)
            print "%d"%(int(statement_id)+1) +"& ", convertToOldInput(unrollPermutations(statement))
            numPermutationsTotal = len(unrollPermutations(statement))

            # 0) remove old files
            removeOldFiles()

            # 1) Generate Code
            print "Generating %d ..."%(statement_id+1)
            cpp_fp = open("transpose.cpp","w+r")
            h_fp = open("transpose.h","w")
            h_fp.write("#include <immintrin.h>\n")
            h_fp.write("#include <omp.h>\n")
            cpp_fp.write("#include \"transpose.h\"\n")
            cpp_fp.write("#include \"transpose_intern.h\"\n")
            # defines
            code_h = "#define BLOCKING2 (blocking*blocking)\n"
            code_h += "#define BLOCKING3 (BLOCKING2*blocking)\n"
            code_h += "#define BLOCKING4 (BLOCKING2*BLOCKING2)\n"
            if ( dim == 3 ):
                code_h += "#define IDXA(a,b,c) (a + b * lda0 + c * lda1)\n"
                code_h += "#define IDXB(a,b,c) (a + b * ldb0 + c * ldb1)\n"
                code_h += "#define IDXW_OUTER(a,b) (a * BLOCKING3 + b * BLOCKING3 * %d)\n"%hpc_implementation.getMaxBlocks(dim)
            else:
                code_h += "#define IDXA(a,b,c,d) (a + b * lda0 + c * lda1 + d * lda2)\n"
                code_h += "#define IDXB(a,b,c,d) (a + b * ldb0 + c * ldb1 + d * ldb2)\n"
                code_h += "#define IDXW_OUTER(a,b) (a * BLOCKING4 + b * BLOCKING4 * %d)\n"%hpc_implementation.getMaxBlocks(dim)
            h_fp.write(code_h)
            cpp_fp.write(getTrashCache("double"))

            reference.genNaiveTempAB(statement, h_fp, cpp_fp)
            reference.genNaiveSpatialB(statement, h_fp, cpp_fp)
            reference.genNaiveSpatialB(statement, h_fp, cpp_fp,1)
            reference.genNaiveMinFlops(statement, h_fp, cpp_fp)
            reference.genNaiveMinFlops(statement_reg, h_fp, cpp_fp, 1)
            reference.genNaiveMinFlopsTempAB(statement, h_fp, cpp_fp)
            reference.genNaiveMinFlopsTempABFused(statement, h_fp, cpp_fp)
            reference.genNaiveMinFlopsTempABFused(statement, h_fp, cpp_fp, 1)

            remainder = -1 # generate all remainders
            if ( not generateOnly ):
                remainder = size % blocking # only generate this remainder to speedup the compilation time

            hpcFunctionName = "spinSummation%d"%(statement_id+1)
            if ( useScalarVersion ):
                hpc_implementation.genHPC_wrapper(statement, h_fp, cpp_fp,
                        blocking, 1, 0, 0, remainder, disableCacheOpt, hpcFunctionName ) # corresponds to Algorithm 5 of paper
            else:
                hpc_implementation.genHPC_wrapper(statement_reg, h_fp, cpp_fp,
                        blocking, useScalarVersion, usePrefetching,
                        useStreamingStores, remainder, disableCacheOpt, hpcFunctionName )
            cpp_fp.close()
            h_fp.close()

            useMinFlopsReg = 0
            useMinFlops = 0
            useTempAB = 0
            useSpatialB = 0
            useMinFlopsTempAB = 0
            useMinFlopsTempABFused = 0
            useHPC = 1
            main.printMain(numPermutationsTotal, "double", size, dim, inplace,
                    numThreads, useMinFlopsReg, useMinFlops, useTempAB,
                    useSpatialB, useMinFlopsTempAB, useMinFlopsTempABFused,
                    useHPC, hpcFunctionName )

            if ( generateOnly ):
                targetDirectory = "spinSummation%d"%(statement_id+1)
                if os.path.exists(targetDirectory):
                    shutil.rmtree(targetDirectory)
                os.mkdir(targetDirectory)
                shutil.copy("Makefile","./%s/"%targetDirectory)
                for filename in glob.glob(r'./*.cpp'):
                    shutil.copy(filename,"./%s/"%targetDirectory)
                for filename in glob.glob(r'./*.h'):
                    shutil.copy(filename,"./%s/"%targetDirectory)
                removeOldFiles()
                continue

            # 2) compile code
            numThreadsCompile = max(2, multiprocessing.cpu_count()/2)
            print "Compiling ..."
            start_time = time.time()
            ret = subprocess.call(["make", "-j%d"%numThreadsCompile], stdout=DEVNULL, stderr=subprocess.STDOUT)
            if ret != 0 :
                print FAIL+"[Error] compilation failed." + ENDC
                exit(-1)
            print "compile time: %.1f sec"%(time.time() - start_time)

            # 3) run code
            print "Running ..."
            start_time = time.time()
            my_env = os.environ.copy()
            my_env["OMP_NUM_THREADS"] = str(numThreads)
            my_env["KMP_AFFINITY"] = affinity 
            proc = subprocess.Popen(['./transpose_sum.exe'],stderr=subprocess.STDOUT,stdout=subprocess.PIPE, env=my_env)

            while True:
                line = proc.stdout.readline()
                line = line.lower()
                if(line.find("_end_") != -1 ):
                    break
                if(line.find("error") != -1 ):
                    print FAIL+"[Error] runtime error." + ENDC
                    exit(-1)
                if(line.find("regminflops:") != -1 ):
                    measurements_minFlopsReg_flops[size_id].append(float(line.split()[5]))
                    measurements_minFlopsReg_minMetric[size_id].append(float(line.split()[3]))
                    maxBW_minMetric = max(maxBW_minMetric, float(line.split()[3]))
                    measurements_minFlopsReg_maxMetric[size_id].append(float(line.split()[1]))
                    maxBW_maxMetric = max(maxBW_maxMetric, float(line.split()[1]))
                if(line.find("refminflops:") != -1 ):
                    measurements_minFlops_flops[size_id].append(float(line.split()[5]))
                    measurements_minFlops_minMetric[size_id].append(float(line.split()[3]))
                    maxBW_minMetric = max(maxBW_minMetric, float(line.split()[3]))
                    measurements_minFlops_maxMetric[size_id].append(float(line.split()[1]))
                    maxBW_maxMetric = max(maxBW_maxMetric, float(line.split()[1]))
                if(line.find("refminflopstempab:") != -1 and line.find("refminflopstempabfused") == -1):
                    print "nonfused:", float(line.split()[3]), " GiB/s"
                    measurements_minFlopsTempAB_flops[size_id].append(float(line.split()[5]))
                    measurements_minFlopsTempAB_minMetric[size_id].append(float(line.split()[3]))
                    maxBW_minMetric = max(maxBW_minMetric, float(line.split()[3]))
                    measurements_minFlopsTempAB_maxMetric[size_id].append(float(line.split()[1]))
                    maxBW_maxMetric = max(maxBW_maxMetric, float(line.split()[1]))
                if(line.find("refminflopstempabfused:") != -1 ):
                    print "fused:", float(line.split()[3]), " GiB/s"
                    measurements_minFlopsTempABFused_flops[size_id].append(float(line.split()[5]))
                    measurements_minFlopsTempABFused_minMetric[size_id].append(float(line.split()[3]))
                    maxBW_minMetric = max(maxBW_minMetric, float(line.split()[3]))
                    measurements_minFlopsTempABFused_maxMetric[size_id].append(float(line.split()[1]))
                    maxBW_maxMetric = max(maxBW_maxMetric, float(line.split()[1]))
                if(line.find("hpc:") != -1 ):
                    print "hpc:", float(line.split()[3]), " GiB/s"
                    measurements_hpc_flops[size_id].append(float(line.split()[5]))
                    measurements_hpc_minMetric[size_id].append(float(line.split()[3]))
                    maxBW_minMetric = max(maxBW_minMetric, float(line.split()[3]))
                    measurements_hpc_maxMetric[size_id].append(float(line.split()[1]))
                    maxBW_maxMetric = max(maxBW_maxMetric, float(line.split()[1]))
                if(line.find("reftempab:") != -1 ):
                    print "ref:", float(line.split()[3]), " GiB/s"
                    measurements_tempAB_flops[size_id].append(float(line.split()[5]))
                    measurements_tempAB_minMetric[size_id].append(float(line.split()[3]))
                    maxBW_minMetric = max(maxBW_minMetric, float(line.split()[3]))
                    measurements_tempAB_maxMetric[size_id].append(float(line.split()[1]))
                    maxBW_maxMetric = max(maxBW_maxMetric, float(line.split()[1]))
                if(line.find("refspatialb:") != -1 ):
                    print "spatialB:", float(line.split()[3]), " GiB/s"
                    measurements_spatialB_flops[size_id].append(float(line.split()[5]))
                    measurements_spatialB_minMetric[size_id].append(float(line.split()[3]))
                    maxBW_minMetric = max(maxBW_minMetric, float(line.split()[3]))
                    measurements_spatialB_maxMetric[size_id].append(float(line.split()[1]))
                    maxBW_maxMetric = max(maxBW_maxMetric, float(line.split()[1]))
            proc.wait()
            print "run time: %.1f sec"%(time.time() - start_time)

            removeOldFiles()

    if not os.path.exists("./data"):
        os.makedirs("./data")
    # save old data files
    for filename in glob.glob(r'./data/*.dat'):
        toRemove = "./data/old/" + filename.split('/')[-1]
        if ( os.path.isfile(toRemove ) ):
            os.remove(toRemove)
        shutil.move(filename,"./data/old/")
    # dump to file
    for size_id in range(len(measurements_minFlops_maxMetric)):
        dumpToFile([measurements_spatialB_minMetric[size_id],
            measurements_spatialB_maxMetric[size_id],
            measurements_spatialB_flops[size_id]], "./data/algo1_spatialB_%d.dat"%size_id)

        dumpToFile([measurements_tempAB_minMetric[size_id], 
            measurements_tempAB_maxMetric[size_id],
            measurements_tempAB_flops[size_id]], "./data/algo2_tempAB_%d.dat"%size_id)

        dumpToFile([measurements_minFlops_minMetric[size_id], 
            measurements_minFlops_maxMetric[size_id], 
            measurements_minFlops_flops[size_id]], "./data/algo3_minFlops_%d.dat"%size_id)
        

        dumpToFile([measurements_minFlopsTempAB_minMetric[size_id],
            measurements_minFlopsTempAB_maxMetric[size_id],
            measurements_minFlopsTempAB_flops[size_id]], "./data/algo35_minFlopsTempAB_%d.dat"%size_id)

        dumpToFile([measurements_minFlopsTempABFused_minMetric[size_id],
            measurements_minFlopsTempABFused_maxMetric[size_id],
            measurements_minFlopsTempABFused_flops[size_id]], "./data/algo4_minFlopsTempABFused_%d.dat"%size_id)

        dumpToFile([measurements_hpc_minMetric[size_id],
            measurements_hpc_maxMetric[size_id],
            measurements_hpc_flops[size_id]], "./data/algoHPC_%d.dat"%size_id)

        dumpToFile([measurements_minFlopsReg_minMetric[size_id],
            measurements_minFlopsReg_maxMetric[size_id],
            measurements_minFlopsReg_flops[size_id]], "./data/algoXX_minFlopsReg_%d.dat"%size_id)
        
        ## plot
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #ax.set_axis_bgcolor((248/256., 248/256., 248/256.))
        #ax.set_ylabel('Bandwidth [GiB/s]',fontsize=22)
        #ax.set_xlabel('Test case',fontsize=22)
        #x = range(len(measurements_minFlops[size_id]))
        #minFlopsAlg = ax.plot(x, measurements_minFlops[size_id], label = "minFlops", color='#31a354', clip_on=False, zorder=10,lw=2.6, ls = "dashed")
        #tempABAlg = ax.plot(x, measurements_tempAB[size_id], label = "tempAB", color='#fdbe85', clip_on=False, zorder=10,lw=2.6, ls = "dashed")
        #spatialBAlg = ax.plot(x, measurements_spatialB[size_id], label = "spatialB", color='#e6550d', clip_on=False, zorder=10,lw=2.6, ls = "dashed")

        #ax.legend( (minFlopsAlg[0], tempABAlg[0], spatialBAlg[0]), ('minFlops','tempAB','spatialB'),loc ='upper center', bbox_to_anchor=(0.5, 1.09), handlelength=0.8, fancybox=True, shadow = True, ncol = 3, numpoints = 1,fontsize=22)
        #
        #plt.savefig("summation_%d.pdf"%size_id, bbox_inches='tight', transparent=False)
        #plt.close()


def printHelp():
    print "Code generator for spin summations. Copyright (C) 2017 Paul Springer.\n" 
    print "This program comes with ABSOLUTELY NO WARRANTY; see LICENSE.txt for details."
    print "This is free software, and you are welcome to redistribute it"
    print "under certain conditions; see LICENSE.txt for details.\n"

    print "required arguments:"
    print "   --numThreads=<numThreads>".ljust(60), "number of Threads"
    print ""
    print "optional arguments:"
    print "   --noVec".ljust(60), "disable explicit vectorization"
    print "   --noPrefetching".ljust(60), "disable SW prefetching"
    print "   --noStreamingStores".ljust(60), "disable steramingStores"
    print "   --benchmark".ljust(60), "enable benchmarking (no code generation)"
    print "   --inplace".ljust(60), "generate inplace transpositions"
    print "   --testcase=<int>".ljust(60), "only execute the specified testcase"
    print "   --disableCacheOpt".ljust(60), "disable cache optimization (you should not use this)"
 
_numThreads = 1
_generateOnly = 1
_useScalarVersion = 0
_usePrefetching = 1
_useStreamingStores = 1
_testcase = -1
_inplace = 0
_disableCacheOpt = 0
_allowedArguments = [
"--numThreads","--noVec","--noPrefetching","--noStreamingStores","--benchmark","--testcase","--inplace",
"--disableCacheOpt","--help"]
for arg in sys.argv:
    if( arg == sys.argv[0]): continue

    valid = 0
    for allowed in _allowedArguments:
        if arg.split("=")[0] == allowed:
            valid = 1
            break;

    if(valid == 0):
        printHelp()
        print FAIL + "Error: argument "+arg.split("=")[0] + " not valid." + ENDC
        exit(-1)

    if arg == "--help":
        printHelp()
    if arg == "--inplace":
        _inplace = 1
    if arg == "--disableCacheOpt":
        _disableCacheOpt = 1
    if arg == "--benchmark":
        _generateOnly = 0
    if arg == "--noVec":
        _useScalarVersion = 1
    if arg == "--noPrefetching":
        _usePrefetching = 0
    if arg == "--noStreamingStores":
        _useStreamingStores = 0
    if arg.find("--numThreads=") != -1:
        _numThreads = int(arg.split("=")[1]) 
    if arg.find("--testcase=") != -1:
        _testcase= int(arg.split("=")[1]) 

if( _useScalarVersion ):
    _usePrefetching = 0

print "numThreads: %d"%_numThreads
benchmark(_numThreads, _generateOnly, _useScalarVersion, _usePrefetching,
        _useStreamingStores, _testcase, _inplace, _disableCacheOpt)




