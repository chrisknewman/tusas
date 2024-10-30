Tusas is a general / flexible code for solving coupled systems of nonlinear partial differential equations.
Tusas was originally developed for phasefield simulation of solidification.
In order for Tusas to be effective, the PDEs must be compatible with structured or unstructured Lagrange (nodal) finite element discretizations
and explicit (Euler) or implicit (Euler, Trapezoid, BDF2) temporal discretizations.

The Tusas approach consists of a finite element spatial discretization of the fully-coupled nonlinear system, which
is treated explicitly or implicitly in time with a preconditioned Jacobian-free Newton-Krylov (JFNK) method.
As the JFNK method only requires a residual, from an implementation standpoint, Tusas allows a flexible framework as it only requires the user to implement code for a the residual equation.
The key to efficient implementation of JFNK is effective preconditioning.
As the dominant cost of JFNK is the linear solver, effective preconditioning 
reduces the number of linear solver iterations per Newton iteration.
The preconditioning strategy in Tusas is based on block factorization and algebraic multigrid
that allows an efficient, implicit time integration. 
As such, Tusas allows flexible precondtioning as it only requires the user to implement code for a row of the preconditioning matrix.
In addition, configuration of the nonlinear system and preconditioner can be performed at runtime.

For questions and inquiries a mailing list is available for discussion: tusas-users@googlegroups.com

Google / gmail users can sign in and click the "Join group" button. Otherwise send an email to: tusas-users+subscribe@googlegroups.com

Current release/version is tusas-1.0.0
