#pragma once

/// \file base/falseType.h
///
/// For allowing static ASSERTions with meaningful error messages to be
/// raised for certain template specializations.
///
/// static_ASSERT( false, ... ) does not work - because it does not depend on any of
/// the template paramteers, thus is evaluated by the compiler even if the template
/// specialization is not being called anywhere!

#include <type_traits>

template < typename T >
struct FalseType : std::false_type
{
};
