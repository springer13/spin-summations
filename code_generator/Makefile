SRC=$(wildcard *.cpp)
OBJ=$(SRC:.cpp=.o)

CXX_FLAGS = -O3 -std=c++11 -Wall 

intel: 
	${MAKE} clean 
	${MAKE} intel2
intel2: CXX=icpc
intel2: CXX_FLAGS += -qopenmp -xhost -restrict 
intel2: all

gcc: 
	${MAKE} clean 
	${MAKE} gcc2
gcc2: CXX=g++
gcc2: CXX_FLAGS += -fopenmp -march=native
gcc2: all

all: ${OBJ} ${OBJ_TRANSPOSE}
	${CXX} ${OBJ} ${OBJ_TRANSPOSE} ${CXX_FLAGS} -o transpose_sum.exe ${CXX_LINK} 
run:
	KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./transpose_sum.exe
%.o: %.cpp
	${CXX} ${CXX_FLAGS} ${INCLUDE_PATH} -c $< -o $@ -I${TTC_C_PATH}/include

clean:
	rm -f *.o ${OBJ} ${OBJ_TRANSPOSE} transpose_sum.exe
