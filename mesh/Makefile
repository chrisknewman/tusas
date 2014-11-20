include $(TRILINOS_INSTALL_DIR)/include/Makefile.export.Trilinos

CXX=$(Trilinos_CXX_COMPILER)
CXX_FLAGS=$(Trilinos_CXX_COMPILER_FLAGS)
INCFLAGS = -I$(TRILINOS_INSTALL_DIR)/include -I./include 

Mesh.o:Mesh.C ./include/Mesh.h
	$(CXX) $(CXX_FLAGS) $(INCFLAGS) -c Mesh.C -o Mesh.o $(LDFLAGS)

clean:
	rm -f *.o  

distclean:clean
	rm -f *~ ./include/*~