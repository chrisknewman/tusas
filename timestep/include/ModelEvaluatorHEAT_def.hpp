#ifndef NOX_THYRA_MODEL_EVALUATOR_HEAT_DEF_HPP
#define NOX_THYRA_MODEL_EVALUATOR_HEAT_DEF_HPP

// Thyra support
#include "Thyra_DefaultSpmdVectorSpace.hpp"
#include "Thyra_DefaultSerialDenseLinearOpWithSolveFactory.hpp"
#include "Thyra_DetachedMultiVectorView.hpp"
#include "Thyra_DetachedVectorView.hpp"
#include "Thyra_MultiVectorStdOps.hpp"
#include "Thyra_VectorStdOps.hpp"
#include "Thyra_PreconditionerBase.hpp"
#include "Thyra_MLPreconditionerFactory.hpp"

// Epetra support
#include "Thyra_EpetraThyraWrappers.hpp"
#include "Thyra_get_Epetra_Operator.hpp"
#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_Import.h"
#include "Epetra_CrsGraph.h"
#include "Epetra_CrsMatrix.h"

// local support
#include "preconditioner.hpp"
#include "basis.hpp"
// Nonmember constuctors

template<class Scalar>
Teuchos::RCP<ModelEvaluatorHEAT<Scalar> >
modelEvaluatorHEAT(const Teuchos::RCP<const Epetra_Comm>& comm,
            Mesh *mesh,
            const Scalar dt)
{
  return Teuchos::rcp(new ModelEvaluatorHEAT<Scalar>(comm,mesh,dt));
}

// Constructor

template<class Scalar>
ModelEvaluatorHEAT<Scalar>::
ModelEvaluatorHEAT(const Teuchos::RCP<const Epetra_Comm>& comm,
            Mesh *mesh,
            const Scalar dt) :
  comm_(comm),
  dt_(dt),
  mesh_(mesh),
  showGetInvalidArg_(false)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using ::Thyra::VectorBase;
  typedef ::Thyra::ModelEvaluatorBase MEB;
  typedef Teuchos::ScalarTraits<Scalar> ST;

  TEUCHOS_ASSERT(nonnull(comm_));
  if (comm_->NumProc() != 1) {
    if (comm_->MyPID() == 0) std::cout<<std::endl<<"Only one processor supported"<<std::endl<<std::endl<<std::endl;
    exit(0);
  }

  //const int num_nodes = num_global_elements_ + 1;
  const int num_nodes = mesh_->get_num_nodes();

  // owned space
  x_owned_map_ = rcp(new Epetra_Map(num_nodes,0,*comm_));
  //x_owned_map_->Print(std::cout);
  x_space_ = ::Thyra::create_VectorSpace(x_owned_map_);

  // residual space
  f_owned_map_ = x_owned_map_;
  f_space_ = x_space_;

  x0_ = ::Thyra::createMember(x_space_);
  V_S(x0_.ptr(), ST::zero());

  // Initialize the graph for W CrsMatrix object
  W_graph_ = createGraph();
  P_ = rcp(new Epetra_CrsMatrix(Copy,*W_graph_));
  prec_ = Teuchos::rcp(new preconditioner<Scalar>(P_, comm_));

  MEB::InArgsSetup<Scalar> inArgs;
  inArgs.setModelEvalDescription(this->description());
  inArgs.setSupports(MEB::IN_ARG_x);
  prototypeInArgs_ = inArgs;

  MEB::OutArgsSetup<Scalar> outArgs;
  outArgs.setModelEvalDescription(this->description());
  outArgs.setSupports(MEB::OUT_ARG_f);
  //outArgs.setSupports(MEB::OUT_ARG_W_op);
  outArgs.setSupports(MEB::OUT_ARG_W_prec);
//   outArgs.set_W_properties(DerivativeProperties(
//                  DERIV_LINEARITY_NONCONST
//                  ,DERIV_RANK_FULL
//                  ,true // supportsAdjoint
//                  ));
  prototypeOutArgs_ = outArgs;

  nominalValues_ = inArgs;
  nominalValues_.set_x(x0_);
}

// Initializers/Accessors

template<class Scalar>
Teuchos::RCP<Epetra_CrsGraph>
ModelEvaluatorHEAT<Scalar>::createGraph()
{
  Teuchos::RCP<Epetra_CrsGraph> W_graph;

  // Create the shell for the
  W_graph = Teuchos::rcp(new Epetra_CrsGraph(Copy, *x_owned_map_, 5));

  //nodes are numbered consecutively by nodeid
  for (int i=0; i < mesh_->get_num_nodes(); i++) {
    std::vector<int> column (mesh_->get_nodal_adj(i));

    column.push_back(i);//cn put the diagonal in
    //cn need something here for more than one pde (see 2dstokes ex)
    W_graph->InsertGlobalIndices(i, column.size(), &column[0]);


  }//i

  W_graph->FillComplete();
  //W_graph->Print(std::cout);
  //exit(0);
  return W_graph;
}

template<class Scalar>
void ModelEvaluatorHEAT<Scalar>::set_x0(const Teuchos::ArrayView<const Scalar> &x0_in)
{
#ifdef TEUCHOS_DEBUG
  TEUCHOS_ASSERT_EQUALITY(x_space_->dim(), x0_in.size());
#endif
  Thyra::DetachedVectorView<Scalar> x0(x0_);
  x0.sv().values()().assign(x0_in);
}


template<class Scalar>
void ModelEvaluatorHEAT<Scalar>::setShowGetInvalidArgs(bool showGetInvalidArg)
{
  showGetInvalidArg_ = showGetInvalidArg;
}

template<class Scalar>
void ModelEvaluatorHEAT<Scalar>::
set_W_factory(const Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<Scalar> >& W_factory)
{
  W_factory_ = W_factory;
}

// Public functions overridden from ModelEvaulator


template<class Scalar>
Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> >
ModelEvaluatorHEAT<Scalar>::get_x_space() const
{
  return x_space_;
}


template<class Scalar>
Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> >
ModelEvaluatorHEAT<Scalar>::get_f_space() const
{
  return f_space_;
}


template<class Scalar>
Thyra::ModelEvaluatorBase::InArgs<Scalar>
ModelEvaluatorHEAT<Scalar>::getNominalValues() const
{
  return nominalValues_;
}


template<class Scalar>
Teuchos::RCP<Thyra::LinearOpBase<Scalar> >
ModelEvaluatorHEAT<Scalar>::create_W_op() const
{
  Teuchos::RCP<Epetra_CrsMatrix> W_epetra =
    Teuchos::rcp(new Epetra_CrsMatrix(::Copy,*W_graph_));

  return Thyra::nonconstEpetraLinearOp(W_epetra);
}

template<class Scalar>
Teuchos::RCP< ::Thyra::PreconditionerBase<Scalar> >
ModelEvaluatorHEAT<Scalar>::create_W_prec() const
{

  const Teuchos::RCP<Thyra::LinearOpBase< Scalar > > P_op = prec_;

  Teuchos::RCP<Thyra::DefaultPreconditioner<Scalar> > prec =
    Teuchos::rcp(new Thyra::DefaultPreconditioner<Scalar>(Teuchos::null,P_op));

  return prec;
//  return Teuchos::null;
}

template<class Scalar>
Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<Scalar> >
ModelEvaluatorHEAT<Scalar>::get_W_factory() const
{
  return W_factory_;
}


template<class Scalar>
Thyra::ModelEvaluatorBase::InArgs<Scalar>
ModelEvaluatorHEAT<Scalar>::createInArgs() const
{
  return prototypeInArgs_;
}


// Private functions overridden from ModelEvaulatorDefaultBase


template<class Scalar>
Thyra::ModelEvaluatorBase::OutArgs<Scalar>
ModelEvaluatorHEAT<Scalar>::createOutArgsImpl() const
{
  return prototypeOutArgs_;
}


template<class Scalar>
void ModelEvaluatorHEAT<Scalar>::evalModelImpl(
  const Thyra::ModelEvaluatorBase::InArgs<Scalar> &inArgs,
  const Thyra::ModelEvaluatorBase::OutArgs<Scalar> &outArgs
  ) const
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::rcp_dynamic_cast;

  TEUCHOS_ASSERT(nonnull(inArgs.get_x()));

  //const Thyra::ConstDetachedVectorView<Scalar> x(inArgs.get_x());

  const RCP<Thyra::VectorBase<Scalar> > f_out = outArgs.get_f();
  //const RCP<Thyra::LinearOpBase<Scalar> > W_out = outArgs.get_W_op();
  //const RCP<Thyra::LinearOpBase<Scalar> > W_out;
  const RCP<Thyra::PreconditionerBase<Scalar> > W_prec_out = outArgs.get_W_prec();


  if ( nonnull(f_out) ||  nonnull(W_prec_out) ) {

    // ****************
    // Get the underlying epetra objects
    // ****************

    RCP<Epetra_Vector> f;
    if (nonnull(f_out)) {
      f = Thyra::get_Epetra_Vector(*f_owned_map_,outArgs.get_f());
      //f->Print(std::cout);
    }

    if (nonnull(W_prec_out)) {
      //std::cout<<"nonnull(W_prec_out))"<<std::endl;
      //RCP<Epetra_Operator> M_epetra = Thyra::get_Epetra_Operator(*(W_prec_out->getNonconstRightPrecOp()));
      //M_inv = rcp_dynamic_cast<Epetra_CrsMatrix>(M_epetra);
      //TEUCHOS_ASSERT(nonnull(M_inv));
    }

    if (nonnull(f))
      f->PutScalar(0.0);
    if (nonnull(P_))
      P_->PutScalar(0.0);


    const Epetra_Vector &u = *(Thyra::get_Epetra_Vector(*x_owned_map_,inArgs.get_x()));

    double jac;
    double *xx, *yy;
    double *uu;
    int n_nodes_per_elem;

    Basis *ubasis;

    for(int blk = 0; blk < mesh_->get_num_elem_blks(); blk++){

      n_nodes_per_elem = mesh_->get_num_nodes_per_elem_in_blk(blk);

      switch(n_nodes_per_elem){
	
      case 3 : // linear triangle
	ubasis = new BasisLTri;
	break;
	
      case 4 : // linear quad
	ubasis = new BasisLQuad;
	break;
	
      }
      xx = new double[n_nodes_per_elem];
      yy = new double[n_nodes_per_elem];
      uu = new double[n_nodes_per_elem];

      // Loop Over # of Finite Elements on Processor

      for (int ne=0; ne < mesh_->get_num_elem_in_blk(blk); ne++) {


	for(int k = 0; k < n_nodes_per_elem; k++){
	  
	  int nodeid = mesh_->get_node_id(blk, ne, k);
	  
	  xx[k] = mesh_->get_x(nodeid);
	  yy[k] = mesh_->get_y(nodeid);
	  uu[k] = u[nodeid];  // copy initial guess or old solution into local temp
	  
	}//k

	// Loop Over Gauss Points
	for(int gp=0; gp < ubasis->ngp; gp++) { 

	  // Calculate the basis function at the gauss point

	  ubasis->getBasis(gp, xx, yy, uu);

	  // Loop over Nodes in Element

	  for (int i=0; i< n_nodes_per_elem; i++) {
	    int row = mesh_->get_node_id(blk, ne, i);
	    double dphidx = ubasis->dphidxi[i]*ubasis->dxidx+ubasis->dphideta[i]*ubasis->detadx;
	    double dphidy = ubasis->dphidxi[i]*ubasis->dxidy+ubasis->dphideta[i]*ubasis->detady;
	    if (nonnull(f)) {
	      double x = ubasis->xx;
	      double y = ubasis->yy;
	      
	      double divgradu = ubasis->dudx*dphidx + ubasis->dudy*dphidy;//(grad u,grad phi)
	      double ut = (ubasis->uu)/dt_*ubasis->phi[i];
	      double pi = 3.141592653589793;
	      //double ff = 2.*ubasis->phi[i];
	      double ff = ((1. + 5.*dt_*pi*pi)*sin(pi*x)*sin(2.*pi*y)/dt_)*ubasis->phi[i];	      

	      (*f)[row]  += ubasis->jac * ubasis->wt * (ut + divgradu - ff);
	      //(*f)[row]  += ubasis->jac * ubasis->wt * (divgradu - ff);
	    }
	    // Loop over Trial Functions
	    if (nonnull(P_)) {
	      for(int j=0;j < n_nodes_per_elem; j++) {
		int column = mesh_->get_node_id(blk, ne, j);
		double dtestdx = ubasis->dphidxi[j]*ubasis->dxidx+ubasis->dphideta[j]*ubasis->detadx;
		double dtestdy = ubasis->dphidxi[j]*ubasis->dxidy+ubasis->dphideta[j]*ubasis->detady;
		double divgrad = dtestdx * dphidx + dtestdy * dphidy;
		double phi_t = ubasis->phi[i] * ubasis->phi[j]/dt_;
		double jac = ubasis->jac*ubasis->wt*(phi_t + divgrad);
		//std::cout<<row<<" "<<column<<" "<<jac<<std::endl;
		P_->SumIntoGlobalValues(row, 1, &jac, &column);
	      }//j
	    }

	  }//i
	}//gp
      }//ne

      delete xx, yy, uu;
      
      if (nonnull(f)) {//cn double check the use of notnull throughout
	for ( int j = 0; j < mesh_->get_node_set(1).size(); j++ ){
	  
	  int row = mesh_->get_node_set_entry(1, j);
	  
	  (*f)[row] =
	    u[row] - 0.0; // Dirichlet BC of zero
	  
	}
	for ( int j = 0; j < mesh_->get_node_set(2).size(); j++ ){
	  
	  int row = mesh_->get_node_set_entry(2, j);
	  
	  (*f)[row] =
	    u[row] - 0.0; // Dirichlet BC of zero
	  
	}
	for ( int j = 0; j < mesh_->get_node_set(3).size(); j++ ){
	  
	  int row = mesh_->get_node_set_entry(3, j);
	  
	  (*f)[row] =
	    u[row] - 0.0; // Dirichlet BC of zero
	  
	}
	for ( int j = 0; j < mesh_->get_node_set(0).size(); j++ ){
	  
	  int row = mesh_->get_node_set_entry(0, j);
	  
	  (*f)[row] =
	    u[row] - 0.0; // Dirichlet BC of zero
	  
	}
	
      }

      if (nonnull(P_)) {
	for ( int j = 0; j < mesh_->get_node_set(1).size(); j++ ){
	  
	  int row = mesh_->get_node_set_entry(1, j);
	  //clear row and put 1 on diagonal
	  std::vector<int> column (mesh_->get_nodal_adj(row));
	  std::vector<double> vals (column.size(),0.);
	  column.push_back(row);
	  vals.push_back(1.);
	  P_->ReplaceGlobalValues (row, vals.size(), &vals[0],&column[0] );
	  
	}
	for ( int j = 0; j < mesh_->get_node_set(2).size(); j++ ){
	  
	  int row = mesh_->get_node_set_entry(2, j);
	  
	  std::vector<int> column (mesh_->get_nodal_adj(row));
	  std::vector<double> vals (column.size(),0.);
	  column.push_back(row);
	  vals.push_back(1.);
	  P_->ReplaceGlobalValues (row, vals.size(), &vals[0],&column[0] );
	  
	}
	for ( int j = 0; j < mesh_->get_node_set(3).size(); j++ ){
	  
	  int row = mesh_->get_node_set_entry(3, j);
	  
	  std::vector<int> column (mesh_->get_nodal_adj(row));
	  std::vector<double> vals (column.size(),0.);
	  column.push_back(row);
	  vals.push_back(1.);
	  P_->ReplaceGlobalValues (row, vals.size(), &vals[0],&column[0] );
	  
	}
	for ( int j = 0; j < mesh_->get_node_set(0).size(); j++ ){
	  
	  int row = mesh_->get_node_set_entry(0, j);
	  
	  std::vector<int> column (mesh_->get_nodal_adj(row));
	  std::vector<double> vals (column.size(),0.);
	  column.push_back(row);
	  vals.push_back(1.);
	  P_->ReplaceGlobalValues (row, vals.size(), &vals[0],&column[0] );
	  
	}
	
      
 	P_->FillComplete();
  	//P_->Print(std::cout);
//  	exit(0);
      }

    }//blk

    //P_->FillComplete();
    if (nonnull(W_prec_out)) {
      prec_->ReComputePreconditioner();
    }

#if 0
    Epetra_Vector& x = *node_coordinates_;

    int ierr = 0;

    double xx[2];
    double uu[2];
    Basis basis;

    // Zero out the objects that will be filled
    if (nonnull(f))
      f->PutScalar(0.0);

    if (nonnull(P_))
      P_->PutScalar(0.0);

    // Loop Over # of Finite Elements on Processor
    for (int ne=0; ne < x_owned_map_->NumMyElements()-1; ne++) {
      
      // Loop Over Gauss Points
      for(int gp=0; gp < 2; gp++) {
	// Get the solution and coordinates at the nodes
	xx[0]=x[ne];
	xx[1]=x[ne+1];
	uu[0]=u[ne];
	uu[1]=u[ne+1];
	// Calculate the basis function at the gauss point
	basis.computeBasis(gp, xx, uu);
	
	// Loop over Nodes in Element
	for (int i=0; i< 2; i++) {
	  //int row=x_ghosted_map_->GID(ne+i);
	  int row=ne+i;
	  //printf("Proc=%d GlobalRow=%d LocalRow=%d Owned=%d\n",
	  //     MyPID, row, ne+i,x_owned_map_.MyGID(row));
	  if (x_owned_map_->MyGID(row)) {
	    if (nonnull(f)) {
	      //(*f)[x_owned_map_->LID(x_ghosted_map_->GID(ne+i))]+=
	      (*f)[ne+i]+=
		+basis.wt*basis.dz
		*(
		  (1.0/(basis.dz*basis.dz))*basis.duu*
		  basis.dphide[i]
		  +basis.phi[i]
		  +basis.uu*basis.phi[i]/dt_
		  );
	    }
	    // 	{
	    //           (*f)[x_owned_map_->LID(x_ghosted_map_->GID(ne+i))]+=
	    //         +basis.wt*basis.dz
	    //         *((1.0/(basis.dz*basis.dz))*basis.duu*
	    //           basis.dphide[i]+factor*basis.uu*basis.uu*basis.phi[i]);
	    //         }
	  }
	  // Loop over Trial Functions
	  if (nonnull(P_)) {
	    for(int j=0;j < 2; j++) {
	      //if (x_owned_map_->MyGID(row)) {
	      //int column=x_ghosted_map_->GID(ne+j);
	      int column=ne+j;
	      //if (row == column) {
	      //           double jac = basis.wt*basis.dz*((1.0/(basis.dz*basis.dz))*
	      //                           basis.dphide[j]*basis.dphide[i]
	      //                           +2.0*factor*basis.uu*basis.phi[j]*
	      //                          basis.phi[i]);
	      double jac = basis.wt*basis.dz*(
					      (1.0/(basis.dz*basis.dz))* basis.dphide[j]*basis.dphide[i]
					      +basis.phi[j]*basis.phi[i]/dt_
					      );
	      ierr = P_->SumIntoGlobalValues(row, 1, &jac, &column);
	      //}
	      //}
	    }
	  }
	}
      }
    }

    // Insert Boundary Conditions and modify Jacobian and function (F)
    // U(0)=1
    if (nonnull(f))
      (*f)[0]= u[0] - 1.0;

    if (nonnull(P_)) {
      int column=0;
      double jac=1.0;
      ierr = P_->ReplaceGlobalValues(0, 1, &jac, &column);
      column=1;
      jac=0.0;
      ierr = P_->ReplaceGlobalValues(0, 1, &jac, &column);
    }

    P_->FillComplete();
    TEUCHOS_ASSERT(ierr > -1);

//     if (nonnull(f)) {
//       Teuchos::RCP< Epetra_Vector > Xe = Teuchos::rcp(new Epetra_Vector(*get_Epetra_Vector (*x_owned_map_, f_out)));
//       Xe->Print(std::cout);
//     }
#endif
  }
}

//====================================================================


#endif
