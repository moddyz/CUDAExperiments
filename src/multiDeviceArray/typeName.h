#pragma once

/// \file typeName.h
///
/// Utilities for extracting the names of types.

#include <cxxabi.h>
#include <string>

template <typename T>
std::string DemangledTypeName()
{
    int   status;
    char* demangled = abi::__cxa_demangle( typeid( T ).name(), 0, 0, &status );
    ASSERT( status );
    std::string typeName = demangled;
    free( demangled );
    return typeName;
}
