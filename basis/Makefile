include $(TRILINOS_INSTALL_DIR)/include/Makefile.export.Trilinos

CXX=$(Trilinos_CXX_COMPILER)
CXX_FLAGS=$(Trilinos_CXX_COMPILER_FLAGS)
INCFLAGS = -I$(TRILINOS_INSTALL_DIR)/include -I./include 

basis.o:basis.cpp
	$(CXX) $(CXX_FLAGS) $(INCFLAGS) -c basis.cpp -o basis.o $(LDFLAGS)

clean:
	rm -f *.o

distclean:clean
	rm -f *~ ./include/*~
