#ifndef DEGREE_H
#define DEGREE_H

#include <cassert>
#include "vprogram/vertex_program.h"


/*
 * Calculate in-degrees for a directed graph.
 * NOTE: For out-degrees, reverse the input graph.
 */


using deg_t = uint32_t;  // degree type


struct DegState : State
{
  deg_t degree = 0;
  std::string to_string() const { return "{degree: " + std::to_string(degree) + "}"; }
};


template <class ew_t = Empty>  // edge weight
class DegVertex : public VertexProgram<ew_t, Empty, deg_t, DegState>  // <W, M, A, S>
{
public:
  using W = ew_t; using M = Empty; using A = deg_t; using S = DegState;
  using VertexProgram<W, M, A, S>::VertexProgram;  // inherit constructors

  M scatter(const DegState& s) { /*LOG.info("scatter-start\n"); */ return M();}
  A gather(const Edge<W>& edge, const M& msg) { /*LOG.info("gather-start\n"); */return 1;}
  void combine(const A& y1, A& y2) { /*LOG.info("combine-start\n"); */ y2 += y1; /*LOG.info("combine-end\n");*/}
  bool apply(const A& y, DegState& s) { /*LOG.info("apply-start\n"); */ s.degree = y; return true;}
};


#endif
