#include <cstdint>
#include <cstdio>
#ifdef __unix__
#include <dlfcn.h>
#endif

namespace {

using xerbla64_fn = void (*)(const char *, const std::int64_t *);

void *resolve_next_xerbla() {
#if defined(__unix__)
    static bool resolved = false;
    static void *func = nullptr;
    if (!resolved) {
        resolved = true;
        func = dlsym(RTLD_NEXT, "xerbla_");
    }
    return func;
#else
    return nullptr;
#endif
}

}  // namespace

extern "C" void xerbla_(const char *srname, const std::int64_t *info) {
    const std::int64_t value = info ? *info : 0;

    if (auto next = reinterpret_cast<xerbla64_fn>(resolve_next_xerbla())) {
        next(srname, &value);
        return;
    }

    std::fprintf(stderr,
                 " ** On entry to %s parameter number %lld had an illegal value\n",
                 srname ? srname : "<null>",
                 static_cast<long long>(value));
}
