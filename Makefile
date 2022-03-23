programName = SLICE_TEST
sourceName = slice_test.f90
sourceName2 =slice_test.o

FC = nvfortran
OPT = -acc -stdpar -Minfo=accel -fast #NVIDIA compiler options
OBJ = $(sourceName2)

%.o : %.F90
	$(FC) -c -o $@ $^ $(OPT) $(INC)

$(programName) : $(OBJ)
	$(FC) -o $@ $^ $(OPT) $(INC) $(LIB)

clean :
	rm -rf *.o *.mod $(programName) *~ *.out output_data s_*
