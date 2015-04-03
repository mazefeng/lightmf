CC = g++
CFLAGS = -shared -fPIC -Wno-deprecated

all : lightmf-train lightmf-test liblightmf.so
	rm -rf output
	mkdir output output/bin output/include output/lib output/model
	mv lightmf-train output/bin
	mv lightmf-test output/bin
	cp *.h output/include
	mv liblightmf.so output/lib
	rm *.o

lightmf-train : lightmf-train.o SparseMatrix.o LatentFactorModel.o MatrixFactorization.o
	$(CC) -o lightmf-train lightmf-train.o SparseMatrix.o LatentFactorModel.o MatrixFactorization.o -lpthread
lightmf-test : lightmf-test.o SparseMatrix.o LatentFactorModel.o MatrixFactorization.o
	$(CC) -o lightmf-test  lightmf-test.o  SparseMatrix.o LatentFactorModel.o MatrixFactorization.o -lpthread

lightmf-train.o : lightmf-train.cpp SparseMatrix.h MatrixFactorization.h LatentFactorModel.h
	$(CC) $(CFLAGS) -c lightmf-train.cpp
lightmf-test.o : lightmf-test.cpp SparseMatrix.h MatrixFactorization.h LatentFactorModel.h
	$(CC) $(CFLAGS) -c lightmf-test.cpp

liblightmf.so : SparseMatrix.h SparseMatrix.o MatrixFactorization.h MatrixFactorization.o LatentFactorModel.h LatentFactorModel.o
	$(CC) $(CFLAGS) SparseMatrix.o MatrixFactorization.o LatentFactorModel.o -o liblightmf.so
SparseMatrix.o : SparseMatrix.h SparseMatrix.cpp
	$(CC) $(CFLAGS) -c SparseMatrix.cpp
MatrixFactorization.o : MatrixFactorization.h MatrixFactorization.cpp
	$(CC) $(CFLAGS) -c MatrixFactorization.cpp
LatentFactorModel.o : LatentFactorModel.h LatentFactorModel.cpp
	$(CC) $(CFLAGS) -c LatentFactorModel.cpp

clean:
	rm -rf *.o *.so output/*


