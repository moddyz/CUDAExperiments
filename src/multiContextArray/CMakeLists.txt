file(GLOB HEADERS *.h *.cuh)
file(GLOB CPPFILES *.cpp)
get_filename_component(LIBRARY_SUFFIX ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
