#ifndef CARPETX_CARPETX_TRAVERSAL_TRAVERSAL_HXX
#define CARPETX_CARPETX_TRAVERSAL_TRAVERSAL_HXX

#include "internal.hxx"

#include <carpetx.hxx>
#include <carpetx_rk.hxx>

#include <cctk_Schedule.h>

#include <optional>
#include <source_location>

namespace CarpetX {

class traversal_scope_t {
public:
  explicit traversal_scope_t(
      window_t window,
      std::source_location site = std::source_location::current());
  ~traversal_scope_t();
  traversal_scope_t(const traversal_scope_t &) = delete;
  traversal_scope_t &operator=(const traversal_scope_t &) = delete;
};

class tail_bin_scope_t {
public:
  explicit tail_bin_scope_t(
      const char *bin,
      std::source_location site = std::source_location::current());
  ~tail_bin_scope_t();
  tail_bin_scope_t(const tail_bin_scope_t &) = delete;
  tail_bin_scope_t &operator=(const tail_bin_scope_t &) = delete;
};

class recovery_scope_t {
public:
  explicit recovery_scope_t(
      std::source_location site = std::source_location::current());
  ~recovery_scope_t();
  recovery_scope_t(const recovery_scope_t &) = delete;
  recovery_scope_t &operator=(const recovery_scope_t &) = delete;
};

class routine_scope_t {
public:
  explicit routine_scope_t(const cFunctionData *routine);
  ~routine_scope_t();
  routine_scope_t(const routine_scope_t &) = delete;
  routine_scope_t &operator=(const routine_scope_t &) = delete;

private:
  const cFunctionData *const routine;
};

window_t
scheduled_window(const char *caller,
                 std::source_location site = std::source_location::current());

const char *scheduled_tail_bin();

bool in_recovery_traversal();

const cFunctionData *scheduled_routine();

void record_recovered_iteration(
    int iteration, std::source_location site = std::source_location::current());

int recovered_iteration();

void record_integrator(
    const rk_config_t &config,
    std::source_location site = std::source_location::current());

const std::optional<rk_config_t> &integrator();

} // namespace CarpetX

#endif
