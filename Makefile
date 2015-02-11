TRILINOS_INSTALL_DIR=$(TRILINOS_DIR)
export TRILINOS_INSTALL_DIR
include $(TRILINOS_INSTALL_DIR)/include/Makefile.export.Trilinos
#include $(TRILINOS_INSTALL_DIR)/include/Makefile.export.Teuchos
#include $(TRILINOS_INSTALL_DIR)/include/Makefile.export.NOX
#include $(TRILINOS_INSTALL_DIR)/include/Makefile.export.KokkosClassic
#include $(TRILINOS_INSTALL_DIR)/include/Makefile.export.Tpetra
#include $(TRILINOS_INSTALL_DIR)/include/Makefile.export.Epetra
#include $(TRILINOS_INSTALL_DIR)/include/Makefile.export.Belos
LDFLAGS = -L$(TRILINOS_INSTALL_DIR)/lib
LDFLAGS += $(Trilinos_LIBRARIES) $(Trilinos_TPL_LIBRARIES)
INCFLAGS = -I. -I$(TRILINOS_INSTALL_DIR)/include -I./mesh/include -I./basis/include \
	-I./preconditioner/include -I./timestep/include -I./input/include

#INCFLAGS += -I$(TRILINOS_INSTALL_DIR)/packages/kokkos/classic/LinAlg
#INCFLAGS += -I$(TRILINOS_INSTALL_DIR)/packages/kokkos/classic/NodeAPI
CXX=$(Trilinos_CXX_COMPILER)
CXX_FLAGS=$(Trilinos_CXX_COMPILER_FLAGS)

all:	tusas.cpp basis/basis.o mesh/Mesh.o input/ReadInput.o
	$(CXX) $(CXX_FLAGS) $(INCFLAGS) tusas.cpp basis/basis.o mesh/Mesh.o input/ReadInput.o -o tusas.x $(LDFLAGS)

timestep/timestep.o:
	cd timestep && $(MAKE)

basis/basis.o:
	cd basis && $(MAKE)

mesh/Mesh.o:
	cd mesh && $(MAKE)

input/ReadInput.o:
	cd input && $(MAKE)

.PHONY: clean distclean
clean:
	rm -rf *.o *.x *.dSYM
	cd basis && $(MAKE) clean
	cd mesh && $(MAKE) clean
	cd preconditioner && $(MAKE) clean
	cd timestep && $(MAKE) clean
	cd input && $(MAKE) clean

distclean:clean
	rm -rf *~
	cd basis && $(MAKE) distclean
	cd mesh && $(MAKE) distclean
	cd preconditioner && $(MAKE) distclean
	cd timestep && $(MAKE) distclean
	cd input && $(MAKE) distclean
