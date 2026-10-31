/**
 * @file arc_native_pybind.cpp
 * @brief pybind11 module implementation for arc_native_sys.
 */

#include "arc_native_pybind.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#if defined(_WIN32)
ArcMmapContext::ArcMmapContext(const std::string& path, size_t size)
    : path_(path), size_(size) {
    if (size == 0) {
        throw std::invalid_argument("mmap size must be non-zero");
    }

    file_handle_ = CreateFileA(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (file_handle_ == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    mapping_handle_ = CreateFileMappingA(
        file_handle_,
        NULL,
        PAGE_READONLY,
        0,
        0,
        NULL
    );

    if (mapping_handle_ == NULL) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        throw std::runtime_error("Failed to create file mapping for: " + path);
    }

    addr_ = MapViewOfFile(
        mapping_handle_,
        FILE_MAP_READ,
        0,
        0,
        size
    );

    if (addr_ == nullptr) {
        CloseHandle(mapping_handle_);
        CloseHandle(file_handle_);
        mapping_handle_ = NULL;
        file_handle_ = INVALID_HANDLE_VALUE;
        throw std::runtime_error("Failed to map view of file: " + path);
    }
}

void ArcMmapContext::unmap() noexcept {
    if (addr_ != nullptr) {
        UnmapViewOfFile(addr_);
        addr_ = nullptr;
    }
    if (mapping_handle_ != NULL) {
        CloseHandle(mapping_handle_);
        mapping_handle_ = NULL;
    }
    if (file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
    }
}
#else
ArcMmapContext::ArcMmapContext(const std::string& path, size_t size)
    : path_(path), size_(size) {
    if (size == 0) {
        throw std::invalid_argument("mmap size must be non-zero");
    }

    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    addr_ = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd_, 0);
    if (addr_ == MAP_FAILED) {
        close(fd_);
        fd_ = -1;
        addr_ = nullptr;
        throw std::runtime_error("Failed to mmap file: " + path);
    }
}

void ArcMmapContext::unmap() noexcept {
    if (addr_ != nullptr && addr_ != MAP_FAILED) {
        munmap(addr_, size_);
        addr_ = nullptr;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}
#endif

ArcMmapContext::~ArcMmapContext() {
    unmap();
}

ArcMmapContext::ArcMmapContext(ArcMmapContext&& other) noexcept
    : path_(std::move(other.path_)),
      size_(other.size_),
      addr_(other.addr_) {
    other.size_ = 0;
    other.addr_ = nullptr;
#if defined(_WIN32)
    file_handle_ = other.file_handle_;
    mapping_handle_ = other.mapping_handle_;
    other.file_handle_ = INVALID_HANDLE_VALUE;
    other.mapping_handle_ = NULL;
#else
    fd_ = other.fd_;
    other.fd_ = -1;
#endif
}

ArcMmapContext& ArcMmapContext::operator=(ArcMmapContext&& other) noexcept {
    if (this != &other) {
        unmap();
        path_ = std::move(other.path_);
        size_ = other.size_;
        addr_ = other.addr_;
        other.size_ = 0;
        other.addr_ = nullptr;
#if defined(_WIN32)
        file_handle_ = other.file_handle_;
        mapping_handle_ = other.mapping_handle_;
        other.file_handle_ = INVALID_HANDLE_VALUE;
        other.mapping_handle_ = NULL;
#else
        fd_ = other.fd_;
        other.fd_ = -1;
#endif
    }
    return *this;
}

PYBIND11_MODULE(arc_native_sys, m) {
    m.doc() = "High-performance Intel Arc native C++20 mmap weight streaming module";
    m.attr("ARC_NATIVE_ABI_VERSION") = ARC_NATIVE_ABI_VERSION;

    py::class_<TensorEntry>(m, "TensorEntry")
        .def_readonly("ptr", &TensorEntry::ptr)
        .def_readonly("size", &TensorEntry::size);

    py::class_<TensorPtrMap>(m, "TensorPtrMap")
        .def(py::init<>())
        .def("insert", &TensorPtrMap::insert, py::arg("name"), py::arg("ptr"), py::arg("size"))
        .def("get", &TensorPtrMap::get, py::arg("name"), py::return_value_policy::reference)
        .def("__len__", &TensorPtrMap::size);

    py::class_<ArcMmapContext>(m, "ArcMmapContext")
        .def(py::init<std::string, size_t>(), py::arg("path"), py::arg("size"))
        .def("ptr", &ArcMmapContext::ptr)
        .def("size", &ArcMmapContext::size)
        .def("is_closed", &ArcMmapContext::is_closed)
        .def("unmap", &ArcMmapContext::unmap)
        .def("__enter__", [](ArcMmapContext& self) -> ArcMmapContext& { return self; })
        .def("__exit__", [](ArcMmapContext& self, py::args) { self.unmap(); });
}
