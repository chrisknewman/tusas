//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////


#ifndef GREEDY_TIE_BREAK_H
#define GREEDY_TIE_BREAK_H

#include "Tpetra_TieBreak.hpp"

template <typename LocalOrdinal,typename GlobalOrdinal>
class GreedyTieBreak : public Tpetra::Details::TieBreak<LocalOrdinal,GlobalOrdinal> 
{
  
public:
  GreedyTieBreak() { }
  
  virtual bool mayHaveSideEffects() const {
    return true;
  }
  
  virtual std::size_t selectedIndex(GlobalOrdinal /* GID */,
				    const std::vector<std::pair<int,LocalOrdinal> > & pid_and_lid) const
  {
    // always choose index of pair with smallest pid
    const std::size_t numLids = pid_and_lid.size();
    std::size_t idx = 0;
    int minpid = pid_and_lid[0].first;
    std::size_t minidx = 0;
    for (idx = 0; idx < numLids; ++idx) {
      if (pid_and_lid[idx].first < minpid) {
	minpid = pid_and_lid[idx].first;
	minidx = idx;
      }
    }
    return minidx;
  }
};


#endif
