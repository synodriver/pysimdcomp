# cython: language_level=3
# cython: cdivision=True
from libc.stdint cimport uint8_t, uint32_t, int64_t

from simdcomp.backends.cython cimport simdcomp


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

# array.array np.ndarray memoryview begin: at least 128*sizeof(uint32_t)
cpdef inline uint32_t maxbits(const uint32_t[::1] begin) nogil:
    return simdcomp.maxbits(<const uint32_t *>&begin[0])

cpdef inline uint32_t maxbits_length(const uint32_t[::1] in_) nogil:
    return simdcomp.maxbits_length(<const uint32_t *>&in_[0], <uint32_t>in_.shape[0])

