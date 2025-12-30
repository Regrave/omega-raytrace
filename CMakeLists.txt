cmake_minimum_required(VERSION 3.10)
project(omega_raytrace C)

set(CMAKE_C_STANDARD 99)

# Optimization flags
if(MSVC)
    set(CMAKE_C_FLAGS_RELEASE "/O2 /DNDEBUG")
else()
    set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
endif()

# Build shared library
add_library(omega_raytrace SHARED src/raytrace.c)

# Set output name (no 'lib' prefix on Windows)
set_target_properties(omega_raytrace PROPERTIES
    PREFIX ""
    OUTPUT_NAME "omega_raytrace"
)

# Link math library on Linux
if(UNIX AND NOT APPLE)
    target_link_libraries(omega_raytrace m)
endif()
