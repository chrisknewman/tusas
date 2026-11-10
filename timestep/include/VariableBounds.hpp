//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
/////////////////////////////////////////////////////////////////////////////

#ifndef VARIABLEBOUNDS_HPP
#define VARIABLEBOUNDS_HPP

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Tpetra_Vector.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_CommHelpers.hpp>

/// Bounds on selected equations of a node-interlaced solution vector,
/// ie x[node*numeqs + eqn] must lie in [lower, upper] for each bounded eqn.
/// The check runs on the Tpetra vector's own device (OpenMP, Serial, CUDA, ...),
/// so it does not copy the solution to the host.
class VariableBounds {
public:
  typedef Tpetra::Vector<> vector_type;
  typedef vector_type::device_type device_type;
  typedef device_type::execution_space execution_space;

  VariableBounds() : numeqs_(0) {}

  VariableBounds(const int numeqs,
                 const std::vector<int> &eqns,
                 const std::vector<double> &lower,
                 const std::vector<double> &upper,
                 const std::string &name)
    : numeqs_(numeqs), name_(name),
      eqns_h_(eqns), lower_h_(lower), upper_h_(upper)
  {
    if (eqns.size() != lower.size() || eqns.size() != upper.size())
      throw std::runtime_error(name_ + ": array sizes must match!");
    if (numeqs_ < 1)
      throw std::runtime_error(name_ + ": number of equations must be positive!");
    for (size_t k = 0; k < eqns.size(); k++) {
      if (eqns[k] < 0 || eqns[k] >= numeqs_)
        throw std::runtime_error(name_ + ": Bounded Equation IDs must be in [0, "
                                 + std::to_string(numeqs_) + ")!");
      if (lower[k] > upper[k])
        throw std::runtime_error(name_ + ": Lower Bounds must not exceed Upper Bounds!");
    }

    const size_t n = eqns.size();
    eqns_  = Kokkos::View<int*,    device_type>("bounded eqns",  n);
    lower_ = Kokkos::View<double*, device_type>("lower bounds",  n);
    upper_ = Kokkos::View<double*, device_type>("upper bounds",  n);
    auto eqns_m  = Kokkos::create_mirror_view(eqns_);
    auto lower_m = Kokkos::create_mirror_view(lower_);
    auto upper_m = Kokkos::create_mirror_view(upper_);
    for (size_t k = 0; k < n; k++) {
      eqns_m(k) = eqns[k];
      lower_m(k) = lower[k];
      upper_m(k) = upper[k];
    }
    Kokkos::deep_copy(eqns_, eqns_m);
    Kokkos::deep_copy(lower_, lower_m);
    Kokkos::deep_copy(upper_, upper_m);
  }

  bool empty() const { return eqns_h_.empty(); }
  size_t size() const { return eqns_h_.size(); }
  int eqn(const size_t k) const { return eqns_h_[k]; }
  double lower(const size_t k) const { return lower_h_[k]; }
  double upper(const size_t k) const { return upper_h_[k]; }

  /// Collective. Returns true if all bounded entries of x are within bounds on all procs.
  /// NaN fails the check. If verbose, each proc with a violation prints the first one it found.
  bool check(const vector_type &x, const Teuchos::Comm<int> &comm, const bool verbose) const
  {
    int localOK = 1;
    size_t bad = 0;
    double badval = 0.;
    if (!empty()) {
      const size_t localLength = x.getLocalLength();
      const size_t nb = size();
      const size_t numnodes = localLength / numeqs_;
      const int numeqs = numeqs_;
      //local copies of the views so the device lambda does not capture this
      const Kokkos::View<const int*,    device_type> eqns  = eqns_;
      const Kokkos::View<const double*, device_type> lower = lower_;
      const Kokkos::View<const double*, device_type> upper = upper_;
      {
        auto x_view = x.getLocalViewDevice(Tpetra::Access::ReadOnly);
        auto x_1d = Kokkos::subview(x_view, Kokkos::ALL(), 0);

        //find the smallest local index of an out of bounds entry
        Kokkos::parallel_reduce("VariableBounds::check",
                                Kokkos::RangePolicy<execution_space>(0, numnodes*nb),
                                KOKKOS_LAMBDA(const size_t i, size_t &minbad) {
            const size_t k = i % nb;
            const size_t lid = (i / nb)*numeqs + eqns(k);
            const double val = x_1d(lid);
            if (!(val >= lower(k) && val <= upper(k)) && lid < minbad) minbad = lid;
          }, Kokkos::Min<size_t>(bad));

        if (bad < localLength) {
          localOK = 0;
          if (verbose) {
            auto badval_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                                Kokkos::subview(x_1d, bad));
            badval = badval_h();
          }
        }
      }
    }

    int globalOK = localOK;
    Teuchos::reduceAll<int, int>(comm, Teuchos::REDUCE_MIN, localOK, Teuchos::outArg(globalOK));

    if (verbose && !localOK) {
      const int eqn = bad % numeqs_;
      size_t k = 0;
      while (k < size() && eqns_h_[k] != eqn) k++;
      std::cout << "  " << name_ << ": out of bounds on proc " << comm.getRank()
                << ", local node " << bad / numeqs_ << ", equation " << eqn
                << ": value = " << badval
                << ", bounds = [" << lower_h_[k] << ", " << upper_h_[k] << "]" << std::endl;
    }
    return (1 == globalOK);
  }

private:
  int numeqs_;
  std::string name_;
  std::vector<int> eqns_h_;
  std::vector<double> lower_h_;
  std::vector<double> upper_h_;
  Kokkos::View<int*,    device_type> eqns_;
  Kokkos::View<double*, device_type> lower_;
  Kokkos::View<double*, device_type> upper_;
};

#endif // VARIABLEBOUNDS_HPP
