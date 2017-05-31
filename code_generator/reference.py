from utils import *
import itertools

def genNaiveTempAB(statements, h_fp, cpp_fp):
    naiveSummation = unrollPermutations(statements) 

    dim = len(statements[0][0][1])
    ldaStr = ""
    for i in range(dim-1):
        ldaStr += "long lda%d, "%i
    # generate header
    header = "void refTempAB(const double* __restrict__ A, %s\n   double* __restrict__ B, %sconst int num)"%(ldaStr, ldaStr.replace('a','b'))
    h_fp.write(header + ";\n")
    code = header + "\n{\n"

    # generate loops
    indentLevel = 1
    indent = "   "
    code += "#pragma omp parallel for schedule(static,1)\n"
    code += "%sfor(int i%d = 0; i%d < num; i%d++){\n"%(indent * indentLevel, dim-1, dim-1, dim-1)
    indentLevel += 1
    for d in reversed(range(dim-1)):
        code += "%sfor(int i%d = 0; i%d < i%d+1; i%d++){\n"%(indent * indentLevel, d,d,d+1,d)
        indentLevel += 1

    # generate update statements
    idxB = []
    for idx in range(dim):
        idxB.append("i%d"%idx)
    for permB in itertools.permutations(range(dim)):
        code += "%sB[IDXB(%s)] = "%(indentLevel * indent, arrayToString(applyPerm(permB,idxB)))
        for (scalar, permA) in naiveSummation:
            code += "%.2f * A[IDXA(%s)] + "%(scalar, arrayToString(applyPerm(permA,applyPerm(permB,idxB))))
        code = code[0:-3]+";\n"

    for d in range(dim):
        indentLevel -= 1
        code += "%s}\n"%(indentLevel * indent)
    code += "}\n"
    cpp_fp.write(code)

def genNaiveSpatialB(statements, h_fp, cpp_fp, useAsHPCremainder = 0):
    naiveSummation = unrollPermutations(statements) 

    dim = len(statements[0][0][1])
    ldaStr = ""
    for i in range(dim-1):
        ldaStr += "long lda%d, "%i
    # generate header
    header = "void refSpatialB(const double* __restrict__ A, %s\n   double* __restrict__ B, %sconst int num)"%(ldaStr, ldaStr.replace('a','b'))
    if ( useAsHPCremainder ):
        header = "void hpc_remainder_spatialB(const double* __restrict__ A, %s\n   double* __restrict__ B, %sconst int num, const int start)"%(ldaStr, ldaStr.replace('a','b'))
    h_fp.write(header + ";\n")
    code = header + "\n{\n"

    # generate loops
    indentLevel = 1
    indent = "   "
    if ( useAsHPCremainder ):
        code += "#pragma omp for schedule(static,1)\n"
    else:
        code += "#pragma omp parallel for schedule(static,1)\n"
    guardTask = ""
    for d in reversed(range(dim)):
        code += "%sfor(int i%d = 0; i%d < num; i%d++){\n"%(indent * indentLevel, d,d,d)
        guardTask += "i%d >= start || "%d
        indentLevel += 1

    if ( useAsHPCremainder ):
        code += "%sif( %s )\n"%(indent * indentLevel, guardTask[0:-4])
    # generate update statements
    idxB = []
    for idx in range(dim):
        idxB.append("i%d"%idx)
    code += "%sB[IDXB(%s)] = "%(indentLevel * indent, arrayToString(idxB))
    for (scalar, permA) in naiveSummation:
        code += "%.2f * A[IDXA(%s)] + "%(scalar, arrayToString(applyPerm(permA,idxB)))
    code = code[0:-3]+";\n"

    for d in range(dim):
        indentLevel -= 1
        code += "%s}\n"%(indentLevel * indent)
    code += "}\n"
    cpp_fp.write(code)

def genNaiveMinFlops(statements, h_fp, cpp_fp, regularized = 0):
    dim = len(statements[0][0][1])
    ldaStr = ""
    for i in range(dim-1):
        ldaStr += "long lda%d, "%i
    # generate header
    header = "void refMinFlops(const double* __restrict__ A, %s\n   double* __restrict__ &B, %sconst int num, double* __restrict__ &work)"%(ldaStr, ldaStr.replace('a','b'))
    if ( regularized ):
        header = header.replace("refMinFlops", "refMinFlops_reg")
    h_fp.write(header + ";\n")
    code = header + "\n{\n"

    inPtr = "A"
    if ( len(statements) == 1 ):
        outPtr = "B"
    else:
        outPtr = "work"
    for statement in statements:
        # generate loops
        indentLevel = 1
        indent = "   "
        code += "#pragma omp parallel for schedule(static,1)\n"
        for d in reversed(range(dim)):
            code += "%sfor(int i%d = 0; i%d < num; i%d++){\n"%(indent * indentLevel, d,d,d)
            indentLevel += 1

        # generate update statements
        idxB = []
        for idx in range(dim):
            idxB.append("i%d"%idx)
        code += "%s%s[IDXB(%s)] = "%(indentLevel * indent, outPtr, arrayToString(idxB))
        for (scalar, permA) in statement:
            code += "%.2f * %s[IDXA(%s)] + "%(scalar, inPtr, arrayToString(applyPerm(permA,idxB)))
        code = code[0:-3]+";\n"

        for d in range(dim):
            indentLevel -= 1
            code += "%s}\n"%(indentLevel * indent)
        inPtr = outPtr
        if ( outPtr == "B" ):
            outPtr = "work"
        else:
            outPtr = "B"
    if ( len(statements) > 1 and len(statements) % 2 != 0 ):
        code += "%sdouble* tmp = B;\n"%(indentLevel * indent)
        code += "%sB = work;\n"%(indentLevel * indent)
        code += "%swork = tmp;\n"%(indentLevel * indent)
    code += "}\n"
    cpp_fp.write(code)

def genNaiveMinFlopsTempAB(statements, h_fp, cpp_fp):
    dim = len(statements[0][0][1])
    ldaStr = ""
    for i in range(dim-1):
        ldaStr += "long lda%d, "%i
    # generate header
    header = "void refMinFlopsTempAB(const double* __restrict__ A, %s\n   double* __restrict__ &B, %sconst int num, double* __restrict__ &work)"%(ldaStr, ldaStr.replace('a','b'))
    h_fp.write(header + ";\n")
    code = header + "\n{\n"

    inPtr = "A"
    if ( len(statements) == 1 ):
        outPtr = "B"
    else:
        outPtr = "work"
    for statement in statements:
        # generate loops
        indentLevel = 1
        indent = "   "
        code += "#pragma omp parallel for schedule(static,1)\n"
        code += "%sfor(int i%d = 0; i%d < num; i%d++){\n"%(indent * indentLevel, dim-1, dim-1, dim-1)
        indentLevel += 1
        for d in reversed(range(dim-1)):
            code += "%sfor(int i%d = 0; i%d < i%d+1; i%d++){\n"%(indent * indentLevel, d,d,d+1,d)
            indentLevel += 1

        # generate update statements
        idxB = []
        for idx in range(dim):
            idxB.append("i%d"%idx)
        for permB in itertools.permutations(range(dim)):
            code += "%s%s[IDXB(%s)] = "%(indentLevel * indent, outPtr, arrayToString(applyPerm(permB,idxB)))
            for (scalar, permA) in statement:
                code += "%.2f * %s[IDXA(%s)] + "%(scalar, inPtr, arrayToString(applyPerm(permA,applyPerm(permB,idxB))))
            code = code[0:-3]+";\n"

        for d in range(dim):
            indentLevel -= 1
            code += "%s}\n"%(indentLevel * indent)
        inPtr = outPtr
        if ( outPtr == "B" ):
            outPtr = "work"
        else:
            outPtr = "B"
    if ( len(statements) > 1 and len(statements) % 2 != 0 ):
        code += "%sdouble* tmp = B;\n"%(indentLevel * indent)
        code += "%sB = work;\n"%(indentLevel * indent)
        code += "%swork = tmp;\n"%(indentLevel * indent)
    code += "}\n"
    cpp_fp.write(code)


def genNaiveMinFlopsTempABFused(statements, h_fp, cpp_fp, useAsHPCremainder = 0):
    dim = len(statements[0][0][1])
    ldaStr = ""
    for i in range(dim-1):
        ldaStr += "long lda%d, "%i
    # generate header
    header = "void refMinFlopsTempABFused(const double* __restrict__ A, %s\n   double* __restrict__ &B, %sconst int num, double* __restrict__ &work)"%(ldaStr, ldaStr.replace('a','b'))
    if( useAsHPCremainder ):
        header = "void hpc_remainder_fused(double* __restrict__ A, %s const int num, const int start)"%(ldaStr)
    h_fp.write(header + ";\n")
    code = header + "\n{\n"

    # generate loops
    indentLevel = 1
    indent = "   "
    if( useAsHPCremainder ):
        code += "#pragma omp for schedule(static,1)\n"
    else:
        code += "#pragma omp parallel for schedule(static,1)\n"
    code += "%sfor(int i%d = 0; i%d < num; i%d++){\n"%(indent * indentLevel, dim-1, dim-1, dim-1)
    indentLevel += 1
    guardTask = ""
    guardTask = "i%d >= start || "%(dim-1)
    for d in reversed(range(dim-1)):
        code += "%sfor(int i%d = 0; i%d < i%d+1; i%d++){\n"%(indent * indentLevel, d,d,d+1,d)
        guardTask += "i%d >= start || "%d
        indentLevel += 1

    if ( useAsHPCremainder ):
        code += "%sif( %s ){\n"%(indent*indentLevel, guardTask[0:-4])
        indentLevel += 1
    declared = {}
    for statement_id in range(len(statements)):
        statement = statements[statement_id]
        # generate update statements
        idxB = []
        for idx in range(dim):
            idxB.append("i%d"%idx)
        for permB in itertools.permutations(range(dim)):
            if ( statement_id == len(statements)-1 ):
                if ( useAsHPCremainder ):
                    code += "%sA[IDXA(%s)] = "%(indentLevel * indent, arrayToString(applyPerm(permB,idxB)))
                else:
                    code += "%sB[IDXB(%s)] = "%(indentLevel * indent, arrayToString(applyPerm(permB,idxB)))
            else:
                variableName = "work%s_%d"%(arrayToString(applyPerm(permB,idxB),1),statement_id%2)
                if ( declared.has_key(variableName) ):
                    declare = ""
                else:
                    declared[variableName] = 1
                    declare = "double "
                code += "%s%s%s = "%(indentLevel * indent, declare, variableName)
            for (scalar, permA) in statement:
                if ( statement_id == 0 ):
                    code += "%.2f * A[IDXA(%s)] + "%(scalar, arrayToString(applyPerm(permA,applyPerm(permB,idxB))))
                else:
                    code += "%.2f * work%s_%d + "%(scalar, arrayToString(applyPerm(permA,applyPerm(permB,idxB)),1),(statement_id-1)%2)
            code = code[0:-3]+";\n"

    if ( useAsHPCremainder ):
        indentLevel -= 1
        code += "%s}\n"%(indentLevel * indent)
    for d in range(dim):
        indentLevel -= 1
        code += "%s}\n"%(indentLevel * indent)

    code += "}\n"
    cpp_fp.write(code)




