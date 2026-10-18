#include <iostream>

#include "pdesolvers.hxx"

#if defined _OPENMP
#include <omp.h>
#elif defined __HIPCC__
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#else
static inline int omp_get_max_threads() { return 1; }
static inline int omp_get_thread_num() { return 0; }
#endif

#include <cctk.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

namespace PDESolvers {

bool csr_t::invariant() const {
  assert(m >= 0);
  assert(n >= 0);
  assert(int(rowptrs.size()) == m + 1);
  assert(int(colvals.size()) == rowptrs.back());
  assert(int(nzvals.size()) == rowptrs.back());
  assert(rowptrs.at(0) == 0);
  for (int i = 0; i < m; ++i) {
    assert(rowptrs.at(i) <= rowptrs.at(i + 1));
    for (int jp = rowptrs.at(i); jp < rowptrs.at(i + 1); ++jp) {
      const int j = colvals.at(jp);
      assert(j >= 0 && j < n);
    }
  }
  return true;
}

void csr_t::insert_element(const int i, const int j, const CCTK_REAL v) {
  if (!(i >= 0 && i < m))
    std::cout << "m=" << m << " n=" << n << " i=" << i << " j=" << j
              << " v=" << v << "\n";
  assert(i >= 0 && i < m);
  assert(j >= 0 && j < n);
  assert(i >= int(rowptrs.size()) - 1);
  while (int(rowptrs.size()) <= i)
    rowptrs.push_back(colvals.size());
  colvals.push_back(j);
  nzvals.push_back(v);
}
void csr_t::finish_inserting() {
  assert(int(rowptrs.size()) <= m);
  while (int(rowptrs.size()) <= m)
    rowptrs.push_back(colvals.size());
}

csr_t::csr_t(
    const int m, const int n,
    const std::vector<
        std::vector<std::vector<std::tuple<int, int, CCTK_REAL> > > > &values)
    : m(m), n(n) {
  for (const auto &values1 : values) {
    for (const auto &values2 : values1) {
      for (const auto &ijv : values2) {
        const int i = std::get<0>(ijv);
        const int j = std::get<1>(ijv);
        const CCTK_REAL v = std::get<2>(ijv);
        insert_element(i, j, v);
      }
    }
  }
  finish_inserting();
  assert(invariant());
}

csr_t::csr_t(const int m, const int n,
             const Arith::spvect<std::tuple<int, int>, CCTK_REAL> &values)
    : m(m), n(n) {
  for (const auto &ijv : values) {
    const int i = std::get<0>(ijv.first);
    const int j = std::get<1>(ijv.first);
    const CCTK_REAL v = ijv.second;
    insert_element(i, j, v);
  }
  finish_inserting();
  assert(invariant());
}

std::size_t csr_t::size() const { return nzvals.size(); }

std::size_t jacobian_t::size() const { return entries.size(); }

void jacobian_t::clear() { entries.clear(); }

void jacobian_t::set_matrix_entries(const csr_t &Jp, Mat J) const {
  std::vector<CCTK_REAL> values;
  for (const auto &e : entries) {
    const auto i = std::get<0>(e);
    const auto j = std::get<1>(e);
    const auto v = std::get<2>(e);
    assert(i >= 0);
    if (j < 0) {
    } else if (j >= 0 && j < prolongation_index_offset) {
      const PetscErrorCode ierr = MatSetValue(J, i, j, v, ADD_VALUES);
      assert(!ierr);
    } else {
      const int row = j - prolongation_index_offset;
      assert(0 <= row && row < Jp.m);
      const int rowptr0 = Jp.rowptrs.at(row);
      const int rowptr1 = Jp.rowptrs.at(row + 1);
      values.resize(rowptr1 - rowptr0);
      for (int rowptr = rowptr0; rowptr < rowptr1; ++rowptr)
        values.at(rowptr - rowptr0) = v * Jp.nzvals.at(rowptr);
      const PetscErrorCode ierr =
          MatSetValues(J, 1, &i, rowptr1 - rowptr0, &Jp.colvals.at(rowptr0),
                       values.data(), ADD_VALUES);
      assert(!ierr);
    }
  }
}

void jacobian_t::set_matrix_entries(
    const csr_t &Jp,
    Arith::spvect<std::tuple<int, int>, CCTK_REAL> &Jsp) const {
  for (const auto &e : entries) {
    const auto i = std::get<0>(e);
    const auto j = std::get<1>(e);
    const auto v = std::get<2>(e);
    assert(i >= 0);
    if (j < 0) {
    } else if (j >= 0 && j < prolongation_index_offset) {
      Jsp.emplace_back(std::make_tuple(i, j), v);
    } else {
      const int row = j - prolongation_index_offset;
      assert(0 <= row && row < Jp.m);
      const int rowptr0 = Jp.rowptrs.at(row);
      const int rowptr1 = Jp.rowptrs.at(row + 1);
      for (int rowptr = rowptr0; rowptr < rowptr1; ++rowptr) {
        const int col = Jp.colvals.at(rowptr);
        const CCTK_REAL val = Jp.nzvals.at(rowptr);
        Jsp.emplace_back(std::make_tuple(i, col), v * val);
      }
    }
  }
}

jacobians_t::jacobians_t() : jacobians(omp_get_max_threads()) {}

jacobian_t &jacobians_t::get_local() {
  return jacobians.at(omp_get_thread_num());
}

std::size_t jacobians_t::size() const {
  std::size_t sz = 0;
  for (auto &j : jacobians)
    sz += j.size();
  return sz;
}

void jacobians_t::clear() {
  for (auto &j : jacobians)
    j.clear();
}

void jacobians_t::define_matrix(const csr_t &Jp, Mat J) const {
  PetscErrorCode ierr;

  PetscBool assembled;
  ierr = MatAssembled(J, &assembled);
  assert(!ierr);
  const bool preallocating = !assembled;
  if (preallocating) {
    MPI_Comm comm;
    ierr = PetscObjectGetComm((PetscObject)J, &comm);
    assert(!ierr);
    PetscInt nlocal_rows, nlocal_cols, nglobal_rows, nglobal_cols;
    ierr = MatGetLocalSize(J, &nlocal_rows, &nlocal_cols);
    assert(!ierr);
    ierr = MatGetSize(J, &nglobal_rows, &nglobal_cols);
    assert(!ierr);

    Mat P;
    ierr = MatCreate(comm, &P);
    assert(!ierr);
    ierr = MatSetType(P, MATPREALLOCATOR);
    assert(!ierr);
    ierr = MatSetSizes(P, nlocal_rows, nlocal_cols, nglobal_rows, nglobal_cols);
    assert(!ierr);
    ierr = MatSetUp(P);
    assert(!ierr);
    for (const auto &j : jacobians)
      j.set_matrix_entries(Jp, P);
    ierr = MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);
    assert(!ierr);
    ierr = MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY);
    assert(!ierr);
    ierr = MatPreallocatorPreallocate(P, PETSC_TRUE, J);
    assert(!ierr);
    ierr = MatDestroy(&P);
    assert(!ierr);
  }

  ierr = MatZeroEntries(J);
  assert(!ierr);
  for (const auto &j : jacobians)
    j.set_matrix_entries(Jp, J);
  ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
  assert(!ierr);
  ierr = MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);
  assert(!ierr);
  ierr = MatSetOption(J, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE);
  assert(!ierr);

  if (preallocating) {
    MatInfo info;
    ierr = MatGetInfo(J, MAT_GLOBAL_SUM, &info);
    assert(!ierr);
    MPI_Comm comm;
    ierr = PetscObjectGetComm((PetscObject)J, &comm);
    assert(!ierr);
    PetscInt rstart, rend;
    ierr = MatGetOwnershipRange(J, &rstart, &rend);
    assert(!ierr);
    PetscInt longest_row = 0;
    for (PetscInt row = rstart; row < rend; ++row) {
      PetscInt ncols;
      ierr = MatGetRow(J, row, &ncols, NULL, NULL);
      assert(!ierr);
      longest_row = std::max(longest_row, ncols);
      ierr = MatRestoreRow(J, row, &ncols, NULL, NULL);
      assert(!ierr);
    }
    ierr =
        MPI_Allreduce(MPI_IN_PLACE, &longest_row, 1, MPIU_INT, MPI_MAX, comm);
    assert(!ierr);
    CCTK_VINFO("Jacobian: %.0f entries used, %.0f allocated (%.1f%% unneeded), "
               "%.0f mallocs, longest row %d",
               double(info.nz_used), double(info.nz_allocated),
               info.nz_allocated > 0
                   ? 100 * double(info.nz_unneeded) / double(info.nz_allocated)
                   : 0.0,
               double(info.mallocs), int(longest_row));
  }
}

std::optional<jacobians_t> jacobians;

} // namespace PDESolvers
