// Manual FP16 (uint bits) -> FP32 conversion

float f16_to_f32(uint h)
{
    h = h & 0xFFFFu;
    uint sgn = (h & 0x8000u) << 16;
    uint exp = (h >> 10) & 0x1Fu;
    uint man = h & 0x3FFu;
    uint f;
    if (exp == 0u) {
        if (man == 0u) f = sgn;
        else {
            exp = 1u;
            while ((man & 0x400u) == 0u) { man <<= 1; exp--; }
            man &= 0x3FFu;
            f = sgn | ((exp + 112u) << 23) | (man << 13);
        }
    } else if (exp == 31u) {
        f = sgn | 0x7F800000u | (man << 13);
    } else {
        f = sgn | ((exp + 112u) << 23) | (man << 13);
    }
    return uintBitsToFloat(f);
}
