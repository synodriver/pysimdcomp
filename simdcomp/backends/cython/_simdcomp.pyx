# cython: language_level=3
# cython: cdivision=True
from libc.stdint cimport uint8_t, uint32_t, int64_t

from simdcomp.backends.cython cimport simdcomp

AVX512BlockSize = simdcomp.AVX512BlockSize
AVXBlockSize = simdcomp.AVXBlockSize
SIMDBlockSize = simdcomp.SIMDBlockSize

cpdef inline uint32_t avx512maxbits(const uint32_t[::1] begin) nogil:
    return simdcomp.avx512maxbits(<const uint32_t *>&begin[0])

cpdef inline avx512pack(const uint32_t[::1] in_, uint32_t[::1] buffer, uint32_t bit):
    with nogil:
        simdcomp.avx512pack(<const uint32_t *>&in_[0], <simdcomp.__m512i *>&buffer[0], bit)

cpdef inline avx512packwithoutmask(const uint32_t[::1] in_, uint32_t[::1] buffer, uint32_t bit):
    with nogil:
        simdcomp.avx512packwithoutmask(<const uint32_t *>&in_[0], <simdcomp.__m512i *>&buffer[0], bit)

cpdef inline avx512unpack(const uint32_t[::1] in_, uint32_t[::1] out, uint32_t bit):
    with nogil:
        simdcomp.avx512unpack(<const simdcomp.__m512i *>&in_[0], &out[0], bit)

cpdef inline uint32_t avxmaxbits(const uint32_t[::1] begin):
    return simdcomp.avxmaxbits(<const uint32_t *>&begin[0])

cpdef inline avxpack(const uint32_t[::1] in_, uint32_t[::1] out, uint32_t bit):
    with nogil:
        simdcomp.avxpack(<const uint32_t *>&in_[0], <simdcomp.__m256i *>&out[0], bit)

cpdef inline avxpackwithoutmask(const uint32_t[::1] in_, uint32_t[::1] out, uint32_t bit):
    with nogil:
        simdcomp.avxpackwithoutmask(<const uint32_t *>&in_[0], <simdcomp.__m256i *>&out[0], bit)

cpdef inline avxunpack(const uint32_t[::1] in_, uint32_t[::1] out, uint32_t bit):
    with nogil:
        simdcomp.avxunpack(<const simdcomp.__m256i *>&in_[0], &out[0], bit)

cpdef inline simdpack(const uint32_t[::1] in_, uint32_t[::1] out, uint32_t bit):
    with nogil:
        simdcomp.simdpack(<const uint32_t *>&in_[0], <simdcomp.__m128i *>&out[0], bit)

cpdef inline simdpackwithoutmask(const uint32_t[::1] in_, uint32_t[::1] out,  uint32_t bit):
    with nogil:
        simdcomp.simdpackwithoutmask(<const uint32_t *>&in_[0],<simdcomp.__m128i *>&out[0], bit)

cpdef inline simdunpack(const uint32_t[::1] in_, uint32_t[::1] out,  uint32_t bit):
    with nogil:
        simdcomp.simdunpack(<const  simdcomp.__m128i *>&in_[0], &out[0], bit)

cpdef inline int simdpack_compressedbytes(int length, uint32_t bit) nogil:
    return simdcomp.simdpack_compressedbytes(length, bit)

cpdef inline int64_t simdpack_length(const uint32_t[::1] in_, uint32_t[::1] out, uint32_t bit):
    cdef simdcomp.__m128i *ret
    with nogil:
        ret = simdcomp.simdpack_length(&in_[0], <size_t>in_.shape[0], <simdcomp.__m128i *>&out[0], bit)
    cdef int64_t buffer_updated = <uint8_t*>ret - <uint8_t*>&out[0]
    return buffer_updated

cpdef inline int64_t simdunpack_length(const uint32_t[::1] in_, uint32_t[::1] out, uint32_t bit):
    cdef simdcomp.__m128i * ret
    with nogil:
        ret = <simdcomp.__m128i *>simdcomp.simdunpack_length(<const simdcomp.__m128i *>&in_[0], <size_t>in_.shape[0], &out[0], bit)
    cdef int64_t buffer_updated = <uint8_t *> ret - <uint8_t *> &out[0]
    return buffer_updated

cpdef inline int64_t simdpack_shortlength(const uint32_t[::1] in_,  uint32_t[::1] out, uint32_t bit):
    cdef simdcomp.__m128i * ret
    with nogil:
        ret = simdcomp.simdpack_shortlength(&in_[0], <int>in_.shape[0],  <simdcomp.__m128i *>&out[0], bit)
    cdef int64_t buffer_updated = <uint8_t *> ret - <uint8_t *> &out[0]
    return buffer_updated

cpdef inline int64_t simdunpack_shortlength(const uint32_t[::1] in_,  uint32_t[::1] out, uint32_t bit):
    cdef const simdcomp.__m128i *ret
    with nogil:
        ret = simdcomp.simdunpack_shortlength(<const simdcomp.__m128i *>&in_[0], <int>in_.shape[0], &out[0], bit)
    cdef int64_t buffer_updated = <uint8_t *> ret - <uint8_t *> &out[0]
    return buffer_updated

cpdef inline simdfastset(const uint32_t[::1] in_, uint32_t bit, uint32_t value, size_t index):
    with nogil:
        simdcomp.simdfastset(<simdcomp.__m128i *>&in_[0], bit, value, index)

cpdef inline uint32_t bits(uint32_t v) nogil:
    return simdcomp.bits(v)

# array.array np.ndarray memoryview begin: at least 128*sizeof(uint32_t)
cpdef inline uint32_t maxbits(const uint32_t[::1] begin) nogil:
    return simdcomp.maxbits(<const uint32_t *>&begin[0])

cpdef inline uint32_t maxbits_length(const uint32_t[::1] in_) nogil:
    return simdcomp.maxbits_length(<const uint32_t *>&in_[0], <uint32_t>in_.shape[0])

cpdef inline uint32_t simdmin(const uint32_t[::1] in_) nogil:
    return simdcomp.simdmin(&in_[0])

cpdef inline uint32_t simdmin_length(const uint32_t[::1] in_) nogil:
    return simdcomp.simdmin_length(&in_[0], <uint32_t>in_.shape[0])

cpdef inline tuple simdmaxmin_length(const uint32_t[::1] in_):
    cdef uint32_t tmin, tmax
    with nogil:
        simdcomp.simdmaxmin_length(&in_[0],<uint32_t>in_.shape[0], &tmin, &tmax)
    return tmin, tmax

cpdef inline tuple simdmaxmin(const uint32_t[::1] in_):
    cdef uint32_t tmin, tmax
    with nogil:
        simdcomp.simdmaxmin(&in_[0],  &tmin, &tmax)
    return tmin, tmax

cpdef inline uint32_t simdmaxbitsd1(const uint32_t[::1] in_, uint32_t initvalue) nogil:
    return simdcomp.simdmaxbitsd1(initvalue, &in_[0])

cpdef inline uint32_t simdmaxbitsd1_length(const uint32_t[::1] in_, uint32_t initvalue) nogil:
    return simdcomp.simdmaxbitsd1_length(initvalue, &in_[0], <uint32_t>in_.shape[0])

cpdef inline simdpackFOR(const uint32_t[::1] in_, uint32_t initvalue, uint32_t[::1] out, uint32_t bit):
    with nogil:
        simdcomp.simdpackFOR(initvalue, &in_[0], <simdcomp.__m128i *>&out[0], bit)

cpdef inline simdunpackFOR(const uint32_t[::1] in_, uint32_t initvalue, uint32_t[::1] out, uint32_t bit):
    with nogil:
        simdcomp.simdunpackFOR(initvalue, <const simdcomp.__m128i *>&in_[0], &out[0], bit)

cpdef inline int simdpackFOR_compressedbytes(int length, uint32_t bit) nogil:
    return simdcomp.simdpackFOR_compressedbytes(length, bit)

cpdef inline int64_t simdpackFOR_length(const uint32_t[::1] in_, uint32_t initvalue, uint32_t[::1] out, uint32_t bit):
    cdef simdcomp.__m128i *ret
    with nogil:
        ret = simdcomp.simdpackFOR_length(initvalue, &in_[0], <int>in_.shape[0], <simdcomp.__m128i *>&out[0], bit)
    cdef int64_t buffer_updated = <uint8_t *> ret - <uint8_t *> &out[0]
    return buffer_updated

cpdef inline int64_t simdunpackFOR_length(const uint32_t[::1] in_, uint32_t initvalue, uint32_t[::1] out, uint32_t bit):
    cdef const simdcomp.__m128i *ret
    with nogil:
        ret = simdcomp.simdunpackFOR_length(initvalue, <const simdcomp.__m128i *>&in_[0], <int>in_.shape[0], &out[0], bit)
    cdef int64_t buffer_updated = <uint8_t *> ret - <uint8_t *> &out[0]
    return buffer_updated

cpdef inline uint32_t simdselectFOR(const uint32_t[::1] in_, uint32_t initvalue, uint32_t bit, int slot) nogil:
    return simdcomp.simdselectFOR(initvalue, <const simdcomp.__m128i *>&in_[0], bit, slot)

cpdef inline simdfastsetFOR(const uint32_t[::1] in_, uint32_t initvalue, uint32_t bit, uint32_t value, size_t index):
    with nogil:
        simdcomp.simdfastsetFOR(initvalue, <simdcomp.__m128i *>&in_[0], bit, value, index)

cpdef inline tuple simdsearchwithlengthFOR(const uint32_t[::1] in_, uint32_t initvalue, uint32_t bit, uint32_t key):
    cdef:
        int ret
        uint32_t presult
    with nogil:
        ret = simdcomp.simdsearchwithlengthFOR(initvalue, <const simdcomp.__m128i *>&in_[0], bit, <int>in_.shape[0], key, &presult)
    return ret, presult