from utils import *

def printMain(numPermutations, floatType, size, dim, inplace,numThreads, useMinFlopsReg, useMinFlops, useTempAB, useSpatialB, useMinFlopsTempAB, useMinFlopsTempABFused, useHPC, hpcFunctionName ):
    ttc_c = 0
    code = "#include <omp.h>\n"
    code += "#include <stdlib.h>\n"
    code += "#include <stdio.h>\n"
    code += "#include <float.h>\n"
    code += "\n"
    code += "//#define PAPI\n"
    code += "#ifdef PAPI\n"
    code += "#include <papi.h>\n"
    code += "#endif\n"
    if(ttc_c):
        code += "#include <ttc_c.h>\n"
    code += "\n"
    code += "#include \"transpose.h\"\n"
    code += "\n"
    code += "\n"
    code += "typedef %s floatType;\n"%floatType
    code += "void trashCache(floatType *A, floatType *B, int n);\n"
    code += "\n"
    code += "int equal(const floatType *A, const floatType*B, size_t* innerSize, size_t* outerSize){\n"
    code += "   int error = 0;\n"
    code += "   const floatType *Atmp= A;\n"
    code += "   const floatType *Btmp= B;\n"
    code += "   //#pragma omp parallel for reduction(+:error)\n"
    indent = "   "
    indentLevel = 1
    offsetStr = "i0 + "
    offset = ""
    errorStr1 = ""
    errorStr2 = ""
    for i in range(dim):
        if( i != 0):
            offset += "outerSize[%d] * "%(i-1)
            offsetStr += "i%d * %s + "%(i,offset[0:-3])
        code += "%sfor(size_t i%d=0;i%d < innerSize[%d]; ++i%d){\n"%(indent * indentLevel, i,i,i,i)
        errorStr1 += "%ld "
        errorStr2 += ",i%d"%i
        indentLevel += 1
    code += "      size_t offset = %s;\n"%(offsetStr[0:-3])
    code += "      floatType Aabs = (Atmp[offset] < 0) ? -Atmp[offset] : Atmp[offset];\n"
    code += "      floatType Babs = (Btmp[offset] < 0) ? -Btmp[offset] : Btmp[offset];\n"
    code += "      floatType max = (Aabs < Babs) ? Babs : Aabs;\n"
    code += "      floatType diff = (Aabs - Babs);\n"
    code += "      diff = (diff < 0) ? -diff : diff;\n"
    code += "      if(diff > 0){\n"
    code += "        floatType relError = ((diff / max) > diff) ? diff : (diff / max); //max of relative and absolute error to avoid problems close to zero\n"
    code += "        if(relError > 1e-4){\n"
    code += "            printf(\"relError: %%.8e %s\\n\",relError%s);\n"%(errorStr1,errorStr2)
    code += "            return 0;\n"
    code += "         }\n"
    code += "      }\n"
    for i in range(dim):
        indentLevel -= 1
        code += "%s}\n"%(indent * indentLevel)
    code += "   return 1;\n"
    code += "}\n"
    code += "\n"
    code += "void restore(const floatType *in, floatType*out, size_t total_size){\n"
    code += "   for(size_t i=0;i < total_size ; ++i){\n"
    code += "      out[i] = in[i];\n"
    code += "   }\n"
    code += "}\n"
    code += "\n"
    code += "int main(int argc, char** argv)\n"
    code += "{\n"
    code += "\n"
    code += "   long long values[3] = { 0, 0, 0};\n"
    code += "#ifdef PAPI\n"
    code += "   int retval, EventSet = PAPI_NULL;\n"
    code += "   floatType alpha = 1.0;\n"
    code += "   floatType beta = 0.0;\n"
    code += "\n"
    code += "   /* Initialize the PAPI library */\n"
    code += "   retval = PAPI_library_init(PAPI_VER_CURRENT);\n"
    code += "\n"
    code += "   if (retval != PAPI_VER_CURRENT) {\n"
    code += "      fprintf(stderr, \"PAPI library init error!\\n\");\n"
    code += "      exit(1);\n"
    code += "   }\n"
    code += "\n"
    code += "   /* Create the Event Set */\n"
    code += "   if (PAPI_create_eventset(&EventSet) != PAPI_OK)\n"
    code += "      printf(\"error\\n\");\n"
    code += "\n"
    code += "   /* Add Total Instructions Executed to our EventSet */\n"
    code += "   if (PAPI_add_event(EventSet, PAPI_TOT_CYC) != PAPI_OK)\n"
    code += "      printf(\"event not supported \\n\");\n"
    code += "   //if (PAPI_add_event(EventSet, PAPI_L1_DCM) != PAPI_OK)\n"
    code += "   if (PAPI_add_event(EventSet, PAPI_L2_DCM) != PAPI_OK) //l2 data cache misses\n"
    code += "      printf(\"event not supported \\n\");\n"
    code += "   if (PAPI_add_event(EventSet, PAPI_L2_DCA) != PAPI_OK) //l2 data cache accesses\n"
    code += "      printf(\"event not supported \\n\");\n"
    code += "#endif\n"
    code += "\n"
    code += "   // allocate\n"
    code += "   int nRepeat = 20;\n"
    code += "   floatType *A, *A_copy, *B, *work, *work2, *B_copy, *B_ref;\n"
    code += "   floatType *trash1, *trash2;\n"
    code += "   size_t largerThanL3 = 1024*1024*42;\n"
    code += "   trash1 = (floatType*) malloc(sizeof(floatType) * largerThanL3);\n"
    code += "   trash2 = (floatType*) malloc(sizeof(floatType) * largerThanL3);\n"
    sizeStr = ""
    innerSizeA = [size for i in range(dim)]
    innerSizeB = innerSizeA
    outerSizeB = innerSizeA
    outerSizeA = innerSizeA
    innerSizeBStr = ""
    outerSizeAStr = ""
    outerSizeBStr = ""
    totalOuterSizeAStr = ""
    totalOuterSizeBStr = ""
    for i in range(dim):
        innerSizeBStr += "%d, "%innerSizeB[i]
        outerSizeAStr += "%d, "%outerSizeA[i]
        outerSizeBStr += "%d, "%outerSizeB[i]
        sizeStr += "%d * "%innerSizeB[i]
        totalOuterSizeAStr += "%d * "%outerSizeA[i]
        totalOuterSizeBStr += "%d * "%outerSizeB[i]
    code += "   size_t innerSizeB[] = { %s };\n"%(innerSizeBStr[0:-2])
    code += "   size_t outerSizeA[] = { %s };\n"%(outerSizeAStr[0:-2])
    code += "   size_t outerSizeB[] = { %s };\n"%(outerSizeBStr[0:-2])
    code += "   int numThreads = %d;\n"%(numThreads)
    code += "   if(argc == 2)\n"
    code += "      numThreads = atoi(argv[1]);\n"
    code += "   else\n"
    code += "      printf(\"Usage: <numThreads>\\n\");\n"
    code += "   printf(\"Using %d threads\\n\", numThreads);\n"
    code += "   size_t total_size_outerA_elem = %s;\n"%(totalOuterSizeAStr[0:-3])
    code += "   size_t total_size_outerB_elem = %s;\n"%(totalOuterSizeBStr[0:-3])
    code += "   size_t total_size_elem = %s;\n"%(sizeStr[0:-3])
    code += "   size_t total_size_bytes = sizeof(floatType) * (total_size_elem);\n"
    code += "   double total_flops = total_size_elem * 2. * %d;\n"%(numPermutations)
    code += "   printf(\"Total memory usage: %e MB\\n\",total_size_bytes/1e6);\n"
    code += "   int ret = posix_memalign((void**) &A, 64, total_size_outerA_elem * sizeof(floatType));\n"
    code += "   ret += posix_memalign((void**) &B, 64, total_size_outerB_elem * sizeof(floatType));\n"
    code += "   ret += posix_memalign((void**) &A_copy, 64, total_size_outerA_elem * sizeof(floatType));\n"
    code += "   ret += posix_memalign((void**) &B_copy, 64, total_size_outerB_elem * sizeof(floatType));\n"
    code += "   ret += posix_memalign((void**) &B_ref, 64, total_size_outerB_elem * sizeof(floatType));\n"
    code += "   ret += posix_memalign((void**) &work, 64, total_size_outerB_elem * sizeof(floatType));\n"
    ldaStr = ""
    for i in range(dim-1):
        ldaStr += "%dl, "%(size**(i+1))
    code += "   size_t workspace = %s(NULL, %s NULL, %s %d, NULL, numThreads); \n"%(hpcFunctionName, ldaStr, ldaStr, size)
    code += "   ret += posix_memalign((void**) &work2, 64, workspace);\n"
    code += "   if( ret != 0){ printf(\"ERROR: posix_memalign failed\\n\"); exit(-1); }\n"
    code += "\n"
    code += "   //initialize A\n"
    code += "#pragma omp parallel for\n"
    code += "   for(int i=0;i < largerThanL3; ++i)\n"
    code += "   {\n"
    code += "      trash1[i] = i*1.0013;\n"
    code += "      trash2[i] = i*1.0043;\n"
    code += "   }\n"
    code += "#pragma omp parallel for \n"
    indent = "   "
    indentLevel = 1
    offsetAStr = "i0 + "
    offsetA = ""
    offsetBStr = "i0 + "
    offsetB = ""
    for i in range(dim):
        if( i != 0):
            offsetA += "outerSizeA[%d] * "%(i-1)
            offsetAStr += "i%d * %s + "%(i,offsetA[0:-3])
            offsetB += "outerSizeB[%d] * "%(i-1)
            offsetBStr += "i%d * %s + "%(i,offsetB[0:-3])
        code += "%sfor(size_t i%d=0;i%d < outerSizeA[%d]; ++i%d){\n"%(indent * indentLevel, i,i,i,i)
        indentLevel += 1
    code += "      size_t offsetA = %s;\n"%(offsetAStr[0:-3])
    code += "      A[offsetA] = (((offsetA+1)*13 % 100) - 50.) / 100.;\n"
    code += "      A_copy[offsetA] = A[offsetA];\n"
    for i in range(dim):
        indentLevel -= 1
        code += "%s}\n"%(indent * indentLevel)

    code += "#pragma omp parallel for \n"
    for i in range(dim):
        code += "%sfor(size_t i%d=0;i%d < outerSizeB[%d]; ++i%d){\n"%(indent * indentLevel, i,i,i,i)
        indentLevel += 1
    code += "      size_t offsetB = %s;\n"%(offsetBStr[0:-3])
    code += "      B[offsetB] = (((offsetB+1)*17 % 100) - 50.) / 100.;\n"
    code += "      B_copy[offsetB] = B[offsetB];\n"
    code += "      B_ref[offsetB] = B[offsetB];\n"
    code += "      work[offsetB] = B[offsetB];\n"
    for i in range(dim):
        indentLevel -= 1
        code += "%s}\n"%(indent * indentLevel)
    code += "\n"
    code += "   unsigned long start, end;\n"
    code += "\n"
    code += "   double my_time = FLT_MAX;\n"
    code += "\n"
    code += "   const double bytes_upper = (1+%d)*total_size_elem*sizeof(floatType)/1024./1024./1024.; \n"%(numPermutations)
    code += "   const double bytes_lower = (1+1)*total_size_elem*sizeof(floatType)/1024./1024./1024.;\n"
    code += "\n"
    code += "   /******************************************\n"
    code += "   * correctness checks \n"
    code += "   ******************************************/\n"
    code += "   refTempAB(A, %s B_ref, %s %d); \n"%(ldaStr, ldaStr, size)
    code += "   refSpatialB(A, %s B, %s %d); \n"%(ldaStr, ldaStr, size)
    code += "   if( !equal(B, B_ref, innerSizeB, outerSizeB) ){\n"
    code += "      printf(\"ERROR in %d\\n\",__LINE__);\n"
    code += "      exit(-1);\n"
    code += "   }  \n"
    code += "   restore(B_copy, B, total_size_outerB_elem);\n"
    code += "   refMinFlops(A, %s B, %s %d, work); \n"%(ldaStr, ldaStr, size)
    code += "   if( !equal(B, B_ref, innerSizeB, outerSizeB) ){\n"
    code += "      printf(\"ERROR in %d\\n\",__LINE__);\n"
    code += "      exit(-1);\n"
    code += "   }  \n"
    code += "   restore(B_copy, B, total_size_outerB_elem);\n"
    code += "   refMinFlops_reg(A, %s B, %s %d, work); \n"%(ldaStr, ldaStr, size)
    code += "   if( !equal(B, B_ref, innerSizeB, outerSizeB) ){\n"
    code += "      printf(\"ERROR in %d\\n\",__LINE__);\n"
    code += "      exit(-1);\n"
    code += "   }  \n"
    code += "   restore(B_copy, B, total_size_outerB_elem);\n"
    code += "   refMinFlopsTempAB(A, %s B, %s %d, work); \n"%(ldaStr, ldaStr, size)
    code += "   if( !equal(B, B_ref, innerSizeB, outerSizeB) ){\n"
    code += "      printf(\"ERROR in %d\\n\",__LINE__);\n"
    code += "      exit(-1);\n"
    code += "   }  \n"
    code += "   restore(B_copy, B, total_size_outerB_elem);\n"
    code += "   refMinFlopsTempABFused(A, %s B, %s %d, work); \n"%(ldaStr, ldaStr, size)
    code += "   if( !equal(B, B_ref, innerSizeB, outerSizeB) ){\n"
    code += "      printf(\"ERROR in %d\\n\",__LINE__);\n"
    code += "      exit(-1);\n"
    code += "   }  \n"
    code += "   restore(B_copy, B, total_size_outerB_elem);\n"
    if( inplace ):
        code += "   %s(A_copy, %s A_copy, %s %d, work2, numThreads); \n"%(hpcFunctionName, ldaStr, ldaStr, size)
        code += "   if( !equal(A_copy, B_ref, innerSizeB, outerSizeB) ){\n"
    else:
        code += "   %s(A, %s B, %s %d, work2, numThreads); \n"%(hpcFunctionName, ldaStr, ldaStr, size)
        code += "   if( !equal(B, B_ref, innerSizeB, outerSizeB) ){\n"
    code += "      printf(\"ERROR in %d\\n\",__LINE__);\n"
    code += "      exit(-1);\n"
    code += "   }  \n"
    code += "   printf(\"All correctness checks succeded.\\n\");\n"
    code += "\n"
    if ( useSpatialB ):
        code += "   /******************************************\n"
        code += "   * refSpatialB\n"
        code += "   ******************************************/\n"
        code += "   values[0] = 0; values[1] = 0; values[2] = 0;\n"
        code += "   my_time = FLT_MAX;\n"
        code += "   for(int i = 0; i < nRepeat; i++)\n"
        code += "   {\n"
        code += "      trashCache(trash1, trash2, largerThanL3);\n"
        code += "#if defined(PAPI)\n"
        code += "      /* Start counting */\n"
        code += "      PAPI_start(EventSet);\n"
        code += "#else\n"
        code += "      double start = omp_get_wtime();\n"
        code += "#endif\n"
        code += "      refSpatialB(A, %s B, %s %d); \n"%(ldaStr, ldaStr, size)
        code += "#if defined(PAPI)\n"
        code += "      PAPI_stop(EventSet, values);\n"
        code += "      double tmp = (*values);\n"
        code += "#else\n"
        code += "      double tmp = omp_get_wtime() - start;\n"
        code += "#endif\n"
        code += "      my_time = (tmp < my_time) ? tmp : my_time;\n"
        code += "   }\n"
        code += "\n"
        code += "   printf(\"refSpatialB:  %.2f GiB/s,  %.2f GiB/s, %.2f GFLOPS/s %.2f sec\\n\", bytes_upper / my_time, bytes_lower / my_time, total_flops / my_time /1e9, my_time );\n"
        code += "\n"
    if ( useTempAB):
        code += "   /******************************************\n"
        code += "   * refTempAB\n"
        code += "   ******************************************/\n"
        code += "   values[0] = 0; values[1] = 0; values[2] = 0;\n"
        code += "   my_time = FLT_MAX;\n"
        code += "   for(int i = 0; i < nRepeat; i++)\n"
        code += "   {\n"
        code += "      trashCache(trash1, trash2, largerThanL3);\n"
        code += "#if defined(PAPI)\n"
        code += "      /* Start counting */\n"
        code += "      PAPI_start(EventSet);\n"
        code += "#else\n"
        code += "      double start = omp_get_wtime();\n"
        code += "#endif\n"
        code += "      refTempAB(A, %s B, %s %d); \n"%(ldaStr, ldaStr, size)
        code += "#if defined(PAPI)\n"
        code += "      PAPI_stop(EventSet, values);\n"
        code += "      double tmp = (*values);\n"
        code += "#else\n"
        code += "      double tmp = omp_get_wtime() - start;\n"
        code += "#endif\n"
        code += "      my_time = (tmp < my_time) ? tmp : my_time;\n"
        code += "   }\n"
        code += "\n"
        code += "   printf(\"reftempAB:  %.2f GiB/s,  %.2f GiB/s, %.2f GFLOPS/s %.2f sec \\n\", bytes_upper / my_time, bytes_lower / my_time, total_flops / my_time /1e9, my_time );\n"
        code += "\n"
    if ( useMinFlops ):
        code += "   /******************************************\n"
        code += "   * refMinFlops\n"
        code += "   ******************************************/\n"
        code += "   values[0] = 0; values[1] = 0; values[2] = 0;\n"
        code += "   my_time = FLT_MAX;\n"
        code += "   for(int i = 0; i < nRepeat; i++)\n"
        code += "   {\n"
        code += "      trashCache(trash1, trash2, largerThanL3);\n"
        code += "#if defined(PAPI)\n"
        code += "      /* Start counting */\n"
        code += "      PAPI_start(EventSet);\n"
        code += "#else\n"
        code += "      double start = omp_get_wtime();\n"
        code += "#endif\n"
        code += "      refMinFlops(A, %s B, %s %d, work); \n"%(ldaStr, ldaStr, size)
        code += "#if defined(PAPI)\n"
        code += "      PAPI_stop(EventSet, values);\n"
        code += "      double tmp = (*values);\n"
        code += "#else\n"
        code += "      double tmp = omp_get_wtime() - start;\n"
        code += "#endif\n"
        code += "      my_time = (tmp < my_time) ? tmp : my_time;\n"
        code += "   }\n"
        code += "\n"
        code += "   printf(\"refMinFlops:  %.2f GiB/s,  %.2f GiB/s, %.2f GFLOPS/s %.2f sec \\n\", bytes_upper / my_time, bytes_lower / my_time, total_flops / my_time /1e9, my_time );\n"
        code += "\n"
    if ( useMinFlopsReg ):
        code += "   /******************************************\n"
        code += "   * refMinFlops_reg\n"
        code += "   ******************************************/\n"
        code += "   values[0] = 0; values[1] = 0; values[2] = 0;\n"
        code += "   my_time = FLT_MAX;\n"
        code += "   for(int i = 0; i < nRepeat; i++)\n"
        code += "   {\n"
        code += "      trashCache(trash1, trash2, largerThanL3);\n"
        code += "#if defined(PAPI)\n"
        code += "      /* Start counting */\n"
        code += "      PAPI_start(EventSet);\n"
        code += "#else\n"
        code += "      double start = omp_get_wtime();\n"
        code += "#endif\n"
        code += "      refMinFlops_reg(A, %s B, %s %d, work); \n"%(ldaStr, ldaStr, size)
        code += "#if defined(PAPI)\n"
        code += "      PAPI_stop(EventSet, values);\n"
        code += "      double tmp = (*values);\n"
        code += "#else\n"
        code += "      double tmp = omp_get_wtime() - start;\n"
        code += "#endif\n"
        code += "      my_time = (tmp < my_time) ? tmp : my_time;\n"
        code += "   }\n"
        code += "\n"
        code += "   printf(\"regMinFlops:  %.2f GiB/s,  %.2f GiB/s, %.2f GFLOPS/s %.2f sec \\n\", bytes_upper / my_time, bytes_lower / my_time, total_flops / my_time /1e9, my_time );\n"
        code += "\n"
    if ( useMinFlopsTempAB ):
        code += "   /******************************************\n"
        code += "   * refMinFlopsTempAB\n"
        code += "   ******************************************/\n"
        code += "   values[0] = 0; values[1] = 0; values[2] = 0;\n"
        code += "   my_time = FLT_MAX;\n"
        code += "   for(int i = 0; i < nRepeat; i++)\n"
        code += "   {\n"
        code += "      trashCache(trash1, trash2, largerThanL3);\n"
        code += "#if defined(PAPI)\n"
        code += "      /* Start counting */\n"
        code += "      PAPI_start(EventSet);\n"
        code += "#else\n"
        code += "      double start = omp_get_wtime();\n"
        code += "#endif\n"
        code += "      refMinFlopsTempAB(A, %s B, %s %d, work); \n"%(ldaStr, ldaStr, size)
        code += "#if defined(PAPI)\n"
        code += "      PAPI_stop(EventSet, values);\n"
        code += "      double tmp = (*values);\n"
        code += "#else\n"
        code += "      double tmp = omp_get_wtime() - start;\n"
        code += "#endif\n"
        code += "      my_time = (tmp < my_time) ? tmp : my_time;\n"
        code += "   }\n"
        code += "\n"
        code += "   printf(\"refMinFlopsTempAB:  %.2f GiB/s,  %.2f GiB/s, %.2f GFLOPS/s %.2f sec \\n\", bytes_upper / my_time, bytes_lower / my_time, total_flops / my_time /1e9, my_time );\n"
        code += "\n"

    if ( useMinFlopsTempABFused ):
        code += "   /******************************************\n"
        code += "   * refMinFlops_reg\n"
        code += "   ******************************************/\n"
        code += "   values[0] = 0; values[1] = 0; values[2] = 0;\n"
        code += "   my_time = FLT_MAX;\n"
        code += "   for(int i = 0; i < nRepeat; i++)\n"
        code += "   {\n"
        code += "      trashCache(trash1, trash2, largerThanL3);\n"
        code += "#if defined(PAPI)\n"
        code += "      /* Start counting */\n"
        code += "      PAPI_start(EventSet);\n"
        code += "#else\n"
        code += "      double start = omp_get_wtime();\n"
        code += "#endif\n"
        code += "      refMinFlopsTempABFused(A, %s B, %s %d, work); \n"%(ldaStr, ldaStr, size)
        code += "#if defined(PAPI)\n"
        code += "      PAPI_stop(EventSet, values);\n"
        code += "      double tmp = (*values);\n"
        code += "#else\n"
        code += "      double tmp = omp_get_wtime() - start;\n"
        code += "#endif\n"
        code += "      my_time = (tmp < my_time) ? tmp : my_time;\n"
        code += "   }\n"
        code += "\n"
        code += "   printf(\"refMinFlopsTempABFused:  %.2f GiB/s,  %.2f GiB/s, %.2f GFLOPS/s %.2f sec \\n\", bytes_upper / my_time, bytes_lower / my_time, total_flops / my_time /1e9, my_time );\n"
        code += "\n"

    if ( useHPC ):
        code += "   /******************************************\n"
        code += "   * refMinFlops_reg\n"
        code += "   ******************************************/\n"
        code += "   values[0] = 0; values[1] = 0; values[2] = 0;\n"
        code += "   my_time = FLT_MAX;\n"
        code += "   for(int i = 0; i < nRepeat; i++)\n"
        code += "   {\n"
        code += "      trashCache(trash1, trash2, largerThanL3);\n"
        code += "#if defined(PAPI)\n"
        code += "      /* Start counting */\n"
        code += "      PAPI_start(EventSet);\n"
        code += "#else\n"
        code += "      double start = omp_get_wtime();\n"
        code += "#endif\n"
        if( inplace ):
            code += "      %s(A_copy, %s A_copy, %s %d, work2, numThreads); \n"%(hpcFunctionName, ldaStr, ldaStr, size)
        else:
            code += "      %s(A, %s B, %s %d, work2, numThreads); \n"%(hpcFunctionName, ldaStr, ldaStr, size)
        code += "#if defined(PAPI)\n"
        code += "      PAPI_stop(EventSet, values);\n"
        code += "      double tmp = (*values);\n"
        code += "#else\n"
        code += "      double tmp = omp_get_wtime() - start;\n"
        code += "#endif\n"
        code += "      my_time = (tmp < my_time) ? tmp : my_time;\n"
        code += "   }\n"
        code += "\n"
        code += "   printf(\"hpc:  %.2f GiB/s,  %.2f GiB/s, %.2f GFLOPS/s %.2f sec \\n\", bytes_upper / my_time, bytes_lower / my_time, total_flops / my_time /1e9, my_time );\n"
        code += "\n"



    code += "//   free(A);\n"
    code += "   printf(\"_END_\");\n"
    code += "   free(work);\n"
    code += "   return 0;\n"
    code += "}\n"
    f = open("main.cpp","w")
    f.write(code)
    f.close()
