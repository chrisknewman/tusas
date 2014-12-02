include $(TRILINOS_INSTALL_DIR)/include/Makefile.export.Trilinos

CXX=$(Trilinos_CXX_COMPILER)
CXX_FLAGS=$(Trilinos_CXX_COMPILER_FLAGS)
INCFLAGS = -I$(TRILINOS_INSTALL_DIR)/include -I./include 

#timestep.o:timestep.cpp
#	$(CXX) $(CXX_FLAGS) $(INCFLAGS) -c timestep.cpp -o timestep.o $(LDFLAGS)

.PHONY: clean distclean
clean:
	rm -f *.o

distclean:clean
	rm -f *~ ./include/*~
