#ifndef CARPETX_CARPETX_UTIL_TASK_MANAGER_HXX
#define CARPETX_CARPETX_UTIL_TASK_MANAGER_HXX

#include "internal.hxx"

#include <functional>
#include <vector>
#include <utility>

namespace CarpetX {

using task_t = std::function<void()>;

class task_manager {
  std::vector<task_t> tasks;

public:
  task_manager();
  ~task_manager();
  void submit(task_t task);
  void submit_serially(task_t task);
  void run_tasks();
  void run_tasks_serially();
};

} // namespace CarpetX

#endif
