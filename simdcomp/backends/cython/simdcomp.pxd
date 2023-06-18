# cython: language_level=3
# cython: cdivision=True
from libc.stdint cimport uint32_t


cdef extern from * nogil:

    ctypedef struct __m256i:
        pass

    ctypedef struct __m256:
        pass

    ctypedef struct __m512i:
        pass

    ctypedef struct __m128i:
        pass

cdef extern from "simdcomp.h" nogil:
    size_t AVX512BlockSize
    size_t AVXBlockSize
    uint32_t avx512maxbits(const uint32_t *begin)

    # reads 512 values from "in", writes  "bit" 512-bit vectors to "out" */
    void avx512pack(const uint32_t *in_, __m512i *out, const uint32_t bit)

    # reads 512 values from "in", writes  "bit" 512-bit vectors to "out" */
    void avx512packwithoutmask(const uint32_t *in_, __m512i *out,
                           const uint32_t bit)

    # reads  "bit" 512-bit vectors from "in", writes  512 values to "out" */
    void avx512unpack(const __m512i *in_, uint32_t *out, const uint32_t bit)


    uint32_t avxmaxbits(const uint32_t *begin)

# reads 256 values from "in", writes  "bit" 256-bit vectors to "out" */
    void avxpack(const uint32_t *in_, __m256i *out, const uint32_t bit)

# reads 256 values from "in", writes  "bit" 256-bit vectors to "out" */
    void avxpackwithoutmask(const uint32_t *in_, __m256i *out, const uint32_t bit)

# reads  "bit" 256-bit vectors from "in", writes  256 values to "out" */
    void avxunpack(const __m256i *in_, uint32_t *out, const uint32_t bit)

    void simdpack(const uint32_t *in_, __m128i *out, const uint32_t bit)

# reads 128 values from "in", writes  "bit" 128-bit vectors to "out".
    # * The input values are assumed to be less than 1<<bit. */
    void simdpackwithoutmask(const uint32_t *in_, __m128i *out, const uint32_t bit)

# reads  "bit" 128-bit vectors from "in", writes  128 values to "out" */
    void simdunpack(const __m128i *in_, uint32_t *out, const uint32_t bit)

# how many compressed bytes are needed to compressed length integers using a
    #bit width of bit with the  simdpack_length function. */
    int simdpack_compressedbytes(int length, const uint32_t bit)

# like simdpack, but supports an undetermined number of inputs.
# * This is useful if you need to unpack an array of integers that is not
# divisible by 128 integers.
# * Returns a pointer to the (advanced) compressed array. Compressed data is
# stored in the memory location between the provided (out) pointer and the
# returned pointer. */
    __m128i *simdpack_length(const uint32_t *in_, size_t length, __m128i *out,
                         const uint32_t bit)

# like simdunpack, but supports an undetermined number of inputs.
# * This is useful if you need to unpack an array of integers that is not
# divisible by 128 integers.
# * Returns a pointer to the (advanced) compressed array. The read compressed
# data is between the provided (in) pointer and the returned pointer. */
    const __m128i *simdunpack_length(const __m128i *in_, size_t length,
                                 uint32_t *out, const uint32_t bit)

# like simdpack, but supports an undetermined small number of inputs. This is
#useful if you need to pack less than 128 integers.
# * Note that this function is much slower.
# * Returns a pointer to the (advanced) compressed array. Compressed data is
#stored in the memory location between the provided (out) pointer and the
#returned pointer. */
    __m128i *simdpack_shortlength(const uint32_t *in_, int length, __m128i *out,
                              const uint32_t bit)

# like simdunpack, but supports an undetermined small number of inputs. This is
# useful if you need to unpack less than 128 integers.
# * Note that this function is much slower.
# * Returns a pointer to the (advanced) compressed array. The read compressed
 #data is between the provided (in) pointer and the returned pointer. */
    const __m128i *simdunpack_shortlength(const __m128i *in_, int length,
                                      uint32_t *out, const uint32_t bit)

# given a block of 128 packed values, this function sets the value at index
 #* "index" to "value" */
    void simdfastset(__m128i *in128, uint32_t b, uint32_t value, size_t index)

    uint32_t bits(const uint32_t v)
    uint32_t maxbits(const uint32_t *begin)
    uint32_t maxbits_length(const uint32_t *in_, uint32_t length)
    size_t SIMDBlockSize
    uint32_t simdmin(const uint32_t *in_)
    uint32_t simdmin_length(const uint32_t *in_, uint32_t length)
    void simdmaxmin_length(const uint32_t *in_, uint32_t length, uint32_t *getmin,
                       uint32_t *getmax)
    void simdmaxmin(const uint32_t *in_, uint32_t *getmin, uint32_t *getmax);
    uint32_t simdmaxbitsd1(uint32_t initvalue, const uint32_t *in_)
    uint32_t simdmaxbitsd1_length(uint32_t initvalue, const uint32_t *in_,
                              uint32_t length)
    void simdpackFOR(uint32_t initvalue, const uint32_t *in_, __m128i *out,
                 const uint32_t bit)
    void simdunpackFOR(uint32_t initvalue, const __m128i *in_, uint32_t *out,
                   const uint32_t bit)
    int simdpackFOR_compressedbytes(int length, const uint32_t bit)

    __m128i *simdpackFOR_length(uint32_t initvalue, const uint32_t *in_, int length,
                            __m128i *out, const uint32_t bit)
    const __m128i *simdunpackFOR_length(uint32_t initvalue, const __m128i *in_,
                                    int length, uint32_t *out,
                                    const uint32_t bit)
    uint32_t simdselectFOR(uint32_t initvalue, const __m128i *in_, uint32_t bit,
                       int slot)

    void simdfastsetFOR(uint32_t initvalue, __m128i *in_, uint32_t bit,
                    uint32_t value, size_t index)
    int simdsearchwithlengthFOR(uint32_t initvalue, const __m128i *in_, uint32_t bit,
                            int length, uint32_t key, uint32_t *presult)
    void simdpackd1(uint32_t initvalue, const uint32_t *in_, __m128i *out,
                const uint32_t bit)
    void simdpackwithoutmaskd1(uint32_t initvalue, const uint32_t *in_, __m128i *out,
                           const uint32_t bit)
    void simdunpackd1(uint32_t initvalue, const __m128i *in_, uint32_t *out,
                  const uint32_t bit)
    int simdsearchd1(__m128i *initOffset, const __m128i *in_, uint32_t bit,
                 uint32_t key, uint32_t *presult)
    int simdsearchwithlengthd1(uint32_t initvalue, const __m128i *in_, uint32_t bit,
                           int length, uint32_t key, uint32_t *presult)
    uint32_t simdselectd1(uint32_t initvalue, const __m128i *in_, uint32_t bit,
                      int slot)
    void simdfastsetd1fromprevious(__m128i *in_, uint32_t bit,
                               uint32_t previousvalue, uint32_t value,
                               size_t index)
    void simdfastsetd1(uint32_t initvalue, __m128i *in_, uint32_t bit,
                   uint32_t value, size_t index)
    void simdscand1(__m128i *initOffset, const __m128i *in_, uint32_t bit)


