#pragma once

struct Mat4f
{
    inline const float& operator()( size_t i_row, size_t i_column ) const
    {
        return m_buffer[ i_row * 4 + i_column ];
    }

    float m_buffer[ 16 ];
};
