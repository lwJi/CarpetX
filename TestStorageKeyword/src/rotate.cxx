#include <cctk.h>
#include <cctk_Arguments.h>

namespace {

int array_size(const cGH *const cctkGH, const char *const group) {
  int size = 0;
  if (CCTK_GroupgshGN(cctkGH, 1, &size, group) != 0)
    CCTK_VERROR("Group '%s': CCTK_GroupgshGN failed", group);
  return size;
}

void fill(const cGH *const cctkGH, const char *const group,
          CCTK_REAL *const data, const CCTK_REAL value) {
  const int size = array_size(cctkGH, group);
  for (int i = 0; i < size; ++i)
    data[i] = value;
}

void expect(const cGH *const cctkGH, const char *const routine,
            const char *const group, const int tl, const CCTK_REAL *const data,
            const CCTK_REAL expected) {
  const int size = array_size(cctkGH, group);
  for (int i = 0; i < size; ++i)
    if (data[i] != expected)
      CCTK_VERROR("%s at iteration %d: group '%s' time level %d element %d "
                  "holds %.17g, expected %.17g",
                  routine, cctkGH->cctk_iteration, group, tl, i,
                  double(data[i]), double(expected));
}

constexpr const char *arr = "TestStorageKeyword::storaged_arr";
constexpr const char *arr3 = "TestStorageKeyword::storaged_arr3";

} // namespace

extern "C" void TestStorageKeyword_Seed(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestStorageKeyword_Seed;

  fill(cctkGH, arr, sa, 0);
  fill(cctkGH, arr, sa_p, -1);
  fill(cctkGH, arr3, sa3, 0);
  fill(cctkGH, arr3, sa3_p, -1);
  fill(cctkGH, arr3, sa3_p_p, -2);
}

extern "C" void TestStorageKeyword_Write(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestStorageKeyword_Write;

  const char *const routine = "TestStorageKeyword_Write";
  const int it = cctk_iteration;
  fill(cctkGH, arr, sa, it);
  fill(cctkGH, arr3, sa3, it);
  expect(cctkGH, routine, arr, 1, sa_p, it - 1);
  expect(cctkGH, routine, arr3, 1, sa3_p, it - 1);
  expect(cctkGH, routine, arr3, 2, sa3_p_p, it - 2);
}

extern "C" void TestStorageKeyword_Read(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestStorageKeyword_Read;

  const char *const routine = "TestStorageKeyword_Read";
  const int it = cctk_iteration;
  expect(cctkGH, routine, arr, 1, sa_p, it - 1);
  expect(cctkGH, routine, arr3, 1, sa3_p, it - 1);
  expect(cctkGH, routine, arr3, 2, sa3_p_p, it - 2);
}
