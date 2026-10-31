/**
 * @file arc_native_pybind.h
 * @brief pybind11 C++20 declarations for ArcMmapContext and TensorPtrMap.
 *
 * Exposes RAII memory management and high-throughput tensor pointer registries
 * under ABI version 4 for the Intel Arc model runner native streaming subsystem.
 */

#ifndef ARC_NATIVE_PYBIND_H
#define ARC_NATIVE_PYBIND_H

#define ARC_NATIVE_ABI_VERSION 4

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <memory>
#include <stdexcept>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

struct TensorEntry {
    uintptr_t ptr;
    size_t size;
};

class TensorPtrMap {
public:
    TensorPtrMap() = default;

    void insert(const std::string& name, uintptr_t ptr, size_t size) {
        entries_[name] = TensorEntry{ptr, size};
    }

    const TensorEntry* get(const std::string& name) const {
        auto it = entries_.find(name);
        if (it != entries_.end()) {
            return &it->second;
        }
        return nullptr;
    }

    size_t size() const noexcept {
        return entries_.size();
    }

private:
    std::unordered_map<std::string, TensorEntry> entries_;
};

class ArcMmapContext {
public:
    ArcMmapContext(const std::string& path, size_t size);
    ~ArcMmapContext();

    ArcMmapContext(const ArcMmapContext&) = delete;
    ArcMmapContext& operator=(const ArcMmapContext&) = delete;

    ArcMmapContext(ArcMmapContext&& other) noexcept;
    ArcMmapContext& operator=(ArcMmapContext&& other) noexcept;

    uintptr_t ptr() const noexcept { return reinterpret_cast<uintptr_t>(addr_); }
    size_t size() const noexcept { return size_; }
    bool is_closed() const noexcept { return addr_ == nullptr; }
    void unmap() noexcept;

private:
    std::string path_;
    size_t size_{0};
    void* addr_{nullptr};
#if defined(_WIN32)
    HANDLE file_handle_{INVALID_HANDLE_VALUE};
    HANDLE mapping_handle_{NULL};
#else
    int fd_{-1};
#endif
};

#endif /* ARC_NATIVE_PYBIND_H */
