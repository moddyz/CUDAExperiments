//
// This file is auto-generated, please do not modify directly!
//

#pragma once

/// \file functions/degrees.h
///
/// Unit conversion from an angle encoded as radians into degrees.

#include <gm/gm.h>

#include <gm/base/constants.h>

GM_NS_OPEN

/// Converts angle \p i_angle from radians to degrees.
///
/// \return the angle in units of degrees.
GM_HOST_DEVICE inline float Degrees( const float& i_angle )
{
    constexpr float radiansToDegreesRatio = 180.0f / GM_PI;
    return i_angle * radiansToDegreesRatio;
}

GM_NS_CLOSE