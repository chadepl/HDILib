SET(CMAKE_EXE_LINKER_FLAGS  "-v")

include_directories(${CMAKE_SOURCE_DIR})
include_directories("/usr/local/opt/flann/include")

# Setup for basic tsne app

add_executable(basic_tsne basic_tsne.cpp)

target_link_libraries(basic_tsne hdidimensionalityreduction)
target_link_libraries(basic_tsne hdidata)
target_link_libraries(basic_tsne hdiutils)

target_link_libraries(basic_tsne "/usr/local/opt/flann/lib/libflann_cpp_s.a")


add_executable(using_eigen basic_eigen.cpp)

add_executable(eigen_tsne eigen_tsne.cpp)

target_link_libraries(eigen_tsne hdidimensionalityreduction)
target_link_libraries(eigen_tsne hdidata)
target_link_libraries(eigen_tsne hdiutils)

target_link_libraries(eigen_tsne "/usr/local/opt/flann/lib/libflann_cpp_s.a")


add_executable(merced_tsne merced_tsne.cpp)

target_link_libraries(merced_tsne hdidimensionalityreduction)
target_link_libraries(merced_tsne hdidata)
target_link_libraries(merced_tsne hdiutils)

target_link_libraries(merced_tsne "/usr/local/opt/flann/lib/libflann_cpp_s.a")


# Notes:
#message(STATUS "llvm library dirs: ${LLVM_LIBRARY_DIRS}")
#message(STATUS "llvm binary dirs: ${LLVM_BINARY_DIR}")
#message(STATUS "llvm include dirs: ${LLVM_INCLUDE_DIRS}")
#message(STATUS "llvm base string: ${LLVM_PACKAGE_VERSION}")