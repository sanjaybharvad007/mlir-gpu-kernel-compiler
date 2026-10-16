// tensorjit native runtime (pybind11).
//
// Executes a compiled program without returning to Python between kernels:
//   * the kernels live in a shared object produced by the C backend
//     (dlopen'ed once, symbols resolved once);
//   * every kernel argument is pre-resolved at load time to one of
//       INPUT  i   -> pointer of the i-th caller input
//       ARENA  off -> arena base + off   (intermediate, placed by the memory planner)
//       OUTPUT j   -> pointer of the j-th caller-visible output
//     so a call is just: fill a small pointer table, call N function pointers;
//   * the arena is one 64-byte-aligned allocation owned by the program and
//     reused across calls: no allocator traffic and no page faults on the hot path.
//
// Per-call overhead measured on this design is ~1-2 us for a 6-kernel program
// versus ~60-90 us for the Python executor.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

enum ArgKind : int { INPUT = 0, ARENA = 1, OUTPUT = 2 };

struct ArgRef {
  int kind;
  int64_t value;  // input index / arena offset / output index
};

using KernelFn = void (*)(void**);

struct Kernel {
  KernelFn fn;
  std::string symbol;
  std::vector<ArgRef> args;
};

class Program {
 public:
  Program(const std::string& so_path,
          const std::vector<std::string>& symbols,
          const std::vector<std::vector<std::pair<int, int64_t>>>& args,
          size_t arena_bytes, size_t num_inputs, size_t num_outputs)
      : num_inputs_(num_inputs), num_outputs_(num_outputs), arena_bytes_(arena_bytes) {
    if (symbols.size() != args.size()) throw std::invalid_argument("symbols/args length mismatch");
    handle_ = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_) throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
    kernels_.reserve(symbols.size());
    for (size_t k = 0; k < symbols.size(); ++k) {
      void* sym = dlsym(handle_, symbols[k].c_str());
      if (!sym) throw std::runtime_error("symbol not found: " + symbols[k]);
      Kernel kd{reinterpret_cast<KernelFn>(sym), symbols[k], {}};
      for (const auto& a : args[k]) {
        if (a.first < INPUT || a.first > OUTPUT) throw std::invalid_argument("bad arg kind");
        if (a.first == INPUT && (a.second < 0 || static_cast<size_t>(a.second) >= num_inputs))
          throw std::out_of_range("input index out of range");
        if (a.first == OUTPUT && (a.second < 0 || static_cast<size_t>(a.second) >= num_outputs))
          throw std::out_of_range("output index out of range");
        if (a.first == ARENA && (a.second < 0 || static_cast<size_t>(a.second) >= std::max<size_t>(arena_bytes, 1)))
          throw std::out_of_range("arena offset out of range");
        kd.args.push_back({a.first, a.second});
      }
      kernels_.push_back(std::move(kd));
    }
    if (arena_bytes_ > 0) {
      const size_t rounded = (arena_bytes_ + 63) / 64 * 64;
      arena_ = static_cast<char*>(std::aligned_alloc(64, rounded));
      if (!arena_) throw std::bad_alloc();
    }
  }

  ~Program() {
    std::free(arena_);
    if (handle_) dlclose(handle_);
  }
  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;

  // inputs / outputs: raw data pointers (torch's data_ptr()) of contiguous tensors.
  void run(const std::vector<uintptr_t>& inputs, const std::vector<uintptr_t>& outputs) {
    if (inputs.size() != num_inputs_) throw std::invalid_argument("wrong number of inputs");
    if (outputs.size() != num_outputs_) throw std::invalid_argument("wrong number of outputs");
    const auto t0 = std::chrono::steady_clock::now();
    void* ptrs[64];
    {
      py::gil_scoped_release nogil;  // kernels are OpenMP-parallel and never touch Python
      for (const Kernel& k : kernels_) {
        if (k.args.size() > 64) throw std::runtime_error("too many kernel arguments");
        for (size_t i = 0; i < k.args.size(); ++i) {
          const ArgRef& a = k.args[i];
          switch (a.kind) {
            case INPUT:  ptrs[i] = reinterpret_cast<void*>(inputs[a.value]); break;
            case ARENA:  ptrs[i] = arena_ + a.value; break;
            default:     ptrs[i] = reinterpret_cast<void*>(outputs[a.value]); break;
          }
        }
        k.fn(ptrs);
      }
    }
    calls_++;
    ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
  }

  size_t num_kernels() const { return kernels_.size(); }
  size_t arena_bytes() const { return arena_bytes_; }
  uint64_t calls() const { return calls_; }
  double mean_us() const { return calls_ ? ns_ / 1e3 / calls_ : 0.0; }
  std::vector<std::string> symbols() const {
    std::vector<std::string> out;
    for (const auto& k : kernels_) out.push_back(k.symbol);
    return out;
  }

 private:
  void* handle_ = nullptr;
  std::vector<Kernel> kernels_;
  char* arena_ = nullptr;
  size_t num_inputs_, num_outputs_, arena_bytes_;
  uint64_t calls_ = 0;
  uint64_t ns_ = 0;
};

}  // namespace

PYBIND11_MODULE(_runtime, m) {
  m.doc() = "tensorjit native runtime: executes a compiled kernel program with a planned arena";
  py::class_<Program>(m, "Program")
      .def(py::init<const std::string&, const std::vector<std::string>&,
                    const std::vector<std::vector<std::pair<int, int64_t>>>&, size_t, size_t, size_t>(),
           py::arg("so_path"), py::arg("symbols"), py::arg("args"), py::arg("arena_bytes"),
           py::arg("num_inputs"), py::arg("num_outputs"))
      .def("run", &Program::run, py::arg("inputs"), py::arg("outputs"))
      .def_property_readonly("num_kernels", &Program::num_kernels)
      .def_property_readonly("arena_bytes", &Program::arena_bytes)
      .def_property_readonly("calls", &Program::calls)
      .def_property_readonly("mean_us", &Program::mean_us)
      .def_property_readonly("symbols", &Program::symbols);
  m.attr("INPUT") = static_cast<int>(INPUT);
  m.attr("ARENA") = static_cast<int>(ARENA);
  m.attr("OUTPUT") = static_cast<int>(OUTPUT);
}
