#define SLANG_PRELUDE_EXPORT

#ifdef __CUDACC_RTC__
#define SLANG_CUDA_RTC 1
#else
#define SLANG_CUDA_RTC 0
#endif

#if SLANG_CUDA_RTC

#else

#include <cstdint>
#include <stdio.h>

#endif

// Define SLANG_CUDA_ENABLE_HALF to use the cuda_fp16 include to add half support.
// For this to work NVRTC needs to have the path to the CUDA SDK.
//
// As it stands the includes paths defined for Slang are passed down to NVRTC. Similarly defines
// defined for the Slang compile are passed down.

#ifdef SLANG_CUDA_ENABLE_HALF
// We don't want half2 operators, because it will implement comparison operators that return a
// bool(!). We want to generate those functions. Doing so means that we will have to define all
// the other half2 operators.
#define __CUDA_NO_HALF2_OPERATORS__
#include <cuda_fp16.h>
#endif

#ifdef SLANG_CUDA_ENABLE_OPTIX
#include <optix.h>
#endif

// Define slang offsetof implementation
#ifndef SLANG_OFFSET_OF
#define SLANG_OFFSET_OF(type, member) (size_t)((char*)&(((type*)0)->member) - (char*)0)
#endif

#ifndef SLANG_ALIGN_OF
#define SLANG_ALIGN_OF(type) __alignof__(type)
#endif

// Must be large enough to cause overflow and therefore infinity
#ifndef SLANG_INFINITY
#define SLANG_INFINITY ((float)(1e+300 * 1e+300))
#endif

// For now we'll disable any asserts in this prelude
#define SLANG_PRELUDE_ASSERT(x)

#ifndef SLANG_CUDA_WARP_SIZE
#define SLANG_CUDA_WARP_SIZE 32
#endif

#define SLANG_CUDA_WARP_MASK \
    (SLANG_CUDA_WARP_SIZE - 1) // Used for masking threadIdx.x to the warp lane index
#define SLANG_CUDA_WARP_BITMASK (~int(0))

//
#define SLANG_FORCE_INLINE inline

#define SLANG_CUDA_CALL __device__

#define SLANG_FORCE_INLINE inline
#define SLANG_INLINE inline


// Since we are using unsigned arithmatic care is need in this comparison.
// It is *assumed* that sizeInBytes >= elemSize. Which means (sizeInBytes >= elemSize) >= 0
// Which means only a single test is needed

// Asserts for bounds checking.
// It is assumed index/count are unsigned types.
#define SLANG_BOUND_ASSERT(index, count) SLANG_PRELUDE_ASSERT(index < count);
#define SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_PRELUDE_ASSERT(index <= (sizeInBytes - elemSize) && (index & 3) == 0);

// Macros to zero index if an access is out of range
#define SLANG_BOUND_ZERO_INDEX(index, count) index = (index < count) ? index : 0;
#define SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    index = (index <= (sizeInBytes - elemSize)) ? index : 0;

// The 'FIX' macro define how the index is fixed. The default is to do nothing. If
// SLANG_ENABLE_BOUND_ZERO_INDEX the fix macro will zero the index, if out of range
#ifdef SLANG_ENABLE_BOUND_ZERO_INDEX
#define SLANG_BOUND_FIX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count) \
    SLANG_BOUND_ZERO_INDEX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#else
#define SLANG_BOUND_FIX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

#ifndef SLANG_BOUND_CHECK
#define SLANG_BOUND_CHECK(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX(index, count)
#endif

#ifndef SLANG_BOUND_CHECK_BYTE_ADDRESS
#define SLANG_BOUND_CHECK_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes)    \
    SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#endif

#ifndef SLANG_BOUND_CHECK_FIXED_ARRAY
#define SLANG_BOUND_CHECK_FIXED_ARRAY(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

// This macro handles how out-of-range surface coordinates are handled;
// I can equal
// cudaBoundaryModeClamp, in which case out-of-range coordinates are clamped to the valid range
// cudaBoundaryModeZero, in which case out-of-range reads return zero and out-of-range writes are
// ignored cudaBoundaryModeTrap, in which case out-of-range accesses cause the kernel execution to
// fail.

#ifndef SLANG_CUDA_BOUNDARY_MODE
#define SLANG_CUDA_BOUNDARY_MODE cudaBoundaryModeZero

// Can be one of SLANG_CUDA_PTX_BOUNDARY_MODE. Only applies *PTX* emitted CUDA operations
// which currently is just RWTextureRW format writes
//
// .trap         causes an execution trap on out-of-bounds addresses
// .clamp        stores data at the nearest surface location (sized appropriately)
// .zero         drops stores to out-of-bounds addresses

#define SLANG_PTX_BOUNDARY_MODE "zero"
#endif

struct TypeInfo
{
    size_t typeSize;
};

template<typename T, size_t SIZE>
struct FixedArray
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }
    SLANG_CUDA_CALL T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }

    T m_data[SIZE];
};

// An array that has no specified size, becomes a 'Array'. This stores the size so it can
// potentially do bounds checking.
template<typename T>
struct Array
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    SLANG_CUDA_CALL T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }

    T* data;
    size_t count;
};

// Typically defined in cuda.h, but we can't ship/rely on that, so just define here
typedef unsigned long long CUtexObject;
typedef unsigned long long CUsurfObject;

// On CUDA sampler state is actually bound up with the texture object. We have a SamplerState type,
// backed as a pointer, to simplify code generation, with the downside that such a binding will take
// up uniform space, even though it will have no effect.
// TODO(JS): Consider ways to strip use of variables of this type so have no binding,
struct SamplerStateUnused;
typedef SamplerStateUnused* SamplerState;


// TODO(JS): Not clear yet if this can be handled on CUDA, by just ignoring.
// For now, just map to the index type.
typedef size_t NonUniformResourceIndex;

// Code generator will generate the specific type
template<typename T, int ROWS, int COLS>
struct Matrix;

typedef int1 bool1;
typedef int2 bool2;
typedef int3 bool3;
typedef int4 bool4;

#if SLANG_CUDA_RTC

typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;
typedef ptrdiff_t intptr_t;

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef size_t uintptr_t;

#endif

typedef long long longlong;
typedef unsigned long long ulonglong;

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

union Union32
{
    uint32_t u;
    int32_t i;
    float f;
};

union Union64
{
    uint64_t u;
    int64_t i;
    double d;
};

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL float make_float(T val)
{
    return (float)val;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float _slang_fmod(float x, float y)
{
    return ::fmodf(x, y);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double _slang_fmod(double x, double y)
{
    return ::fmod(x, y);
}

#if SLANG_CUDA_ENABLE_HALF

// Add the other vector half types
struct __half1
{
    __half x;
};
struct __align__(4) __half3
{
    __half x, y, z;
};
struct __align__(4) __half4
{
    __half x, y, z, w;
};
#endif

#define SLANG_VECTOR_GET_ELEMENT(T)                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##1 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##2 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##3 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##4 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }
SLANG_VECTOR_GET_ELEMENT(int)
SLANG_VECTOR_GET_ELEMENT(uint)
SLANG_VECTOR_GET_ELEMENT(short)
SLANG_VECTOR_GET_ELEMENT(ushort)
SLANG_VECTOR_GET_ELEMENT(char)
SLANG_VECTOR_GET_ELEMENT(uchar)
SLANG_VECTOR_GET_ELEMENT(longlong)
SLANG_VECTOR_GET_ELEMENT(ulonglong)
SLANG_VECTOR_GET_ELEMENT(float)
SLANG_VECTOR_GET_ELEMENT(double)

#define SLANG_VECTOR_GET_ELEMENT_PTR(T)                                                      \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##1 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }                                                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##2 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }                                                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##3 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }                                                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##4 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }
SLANG_VECTOR_GET_ELEMENT_PTR(int)
SLANG_VECTOR_GET_ELEMENT_PTR(uint)
SLANG_VECTOR_GET_ELEMENT_PTR(short)
SLANG_VECTOR_GET_ELEMENT_PTR(ushort)
SLANG_VECTOR_GET_ELEMENT_PTR(char)
SLANG_VECTOR_GET_ELEMENT_PTR(uchar)
SLANG_VECTOR_GET_ELEMENT_PTR(longlong)
SLANG_VECTOR_GET_ELEMENT_PTR(ulonglong)
SLANG_VECTOR_GET_ELEMENT_PTR(float)
SLANG_VECTOR_GET_ELEMENT_PTR(double)

#if SLANG_CUDA_ENABLE_HALF
SLANG_VECTOR_GET_ELEMENT(__half)
SLANG_VECTOR_GET_ELEMENT_PTR(__half)
#endif

#define SLANG_CUDA_VECTOR_BINARY_OP(T, n, op)                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator op(T##n thisVal, T##n other)             \
    {                                                                                         \
        T##n result;                                                                          \
        for (int i = 0; i < n; i++)                                                           \
            *_slang_vector_get_element_ptr(&result, i) =                                      \
                _slang_vector_get_element(thisVal, i) op _slang_vector_get_element(other, i); \
        return result;                                                                        \
    }
#define SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, op)                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL bool##n operator op(T##n thisVal, T##n other) \
    {                                                                                \
        bool##n result;                                                              \
        for (int i = 0; i < n; i++)                                                  \
            *_slang_vector_get_element_ptr(&result, i) =                             \
                (int)(_slang_vector_get_element(thisVal, i)                          \
                          op _slang_vector_get_element(other, i));                   \
        return result;                                                               \
    }
#define SLANG_CUDA_VECTOR_UNARY_OP(T, n, op)                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator op(T##n thisVal)                              \
    {                                                                                              \
        T##n result;                                                                               \
        for (int i = 0; i < n; i++)                                                                \
            *_slang_vector_get_element_ptr(&result, i) = op _slang_vector_get_element(thisVal, i); \
        return result;                                                                             \
    }

#define SLANG_CUDA_VECTOR_INT_OP(T, n)            \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, +)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, -)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, *)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, /)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, %)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ^)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, |)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &&)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ||)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, >>)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, <<)         \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, ==) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, !=) \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, !)           \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, -)           \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, ~)

#define SLANG_CUDA_VECTOR_INT_OPS(T) \
    SLANG_CUDA_VECTOR_INT_OP(T, 2)   \
    SLANG_CUDA_VECTOR_INT_OP(T, 3)   \
    SLANG_CUDA_VECTOR_INT_OP(T, 4)

SLANG_CUDA_VECTOR_INT_OPS(int)
SLANG_CUDA_VECTOR_INT_OPS(uint)
SLANG_CUDA_VECTOR_INT_OPS(ushort)
SLANG_CUDA_VECTOR_INT_OPS(short)
SLANG_CUDA_VECTOR_INT_OPS(char)
SLANG_CUDA_VECTOR_INT_OPS(uchar)
SLANG_CUDA_VECTOR_INT_OPS(longlong)
SLANG_CUDA_VECTOR_INT_OPS(ulonglong)

#define SLANG_CUDA_VECTOR_FLOAT_OP(T, n)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, +)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, -)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, *)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, /)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &&)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ||)         \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, ==) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, !=) \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, -)
#define SLANG_CUDA_VECTOR_FLOAT_OPS(T) \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 2)   \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 3)   \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 4)

SLANG_CUDA_VECTOR_FLOAT_OPS(float)
SLANG_CUDA_VECTOR_FLOAT_OPS(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_CUDA_VECTOR_FLOAT_OPS(__half)
#endif
#define SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, n)                                             \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator%(const T##n& left, const T##n& right) \
    {                                                                                      \
        T##n result;                                                                       \
        for (int i = 0; i < n; i++)                                                        \
            *_slang_vector_get_element_ptr(&result, i) = _slang_fmod(                      \
                _slang_vector_get_element(left, i),                                        \
                _slang_vector_get_element(right, i));                                      \
        return result;                                                                     \
    }
#define SLANG_CUDA_FLOAT_VECTOR_MOD(T)     \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 2) \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 3) \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 4)

SLANG_CUDA_FLOAT_VECTOR_MOD(float)
SLANG_CUDA_FLOAT_VECTOR_MOD(double)

#if SLANG_CUDA_RTC || SLANG_CUDA_ENABLE_HALF
#define SLANG_MAKE_VECTOR(T)                                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x, T y)           \
    {                                                                       \
        return T##2 {x, y};                                                 \
    }                                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x, T y, T z)      \
    {                                                                       \
        return T##3 {x, y, z};                                              \
    }                                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x, T y, T z, T w) \
    {                                                                       \
        return T##4 {x, y, z, w};                                           \
    }
#endif

#if SLANG_CUDA_RTC
SLANG_MAKE_VECTOR(int)
SLANG_MAKE_VECTOR(uint)
SLANG_MAKE_VECTOR(short)
SLANG_MAKE_VECTOR(ushort)
SLANG_MAKE_VECTOR(char)
SLANG_MAKE_VECTOR(uchar)
SLANG_MAKE_VECTOR(float)
SLANG_MAKE_VECTOR(double)
SLANG_MAKE_VECTOR(longlong)
SLANG_MAKE_VECTOR(ulonglong)
#endif

#if SLANG_CUDA_ENABLE_HALF
SLANG_MAKE_VECTOR(__half)
#endif

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool1 make_bool1(bool x)
{
    return bool1{x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool2 make_bool2(bool x, bool y)
{
    return bool2{x, y};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool3 make_bool3(bool x, bool y, bool z)
{
    return bool3{x, y, z};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool4 make_bool4(bool x, bool y, bool z, bool w)
{
    return bool4{x, y, z, w};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool2 make_bool2(bool x)
{
    return bool2{x, x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool3 make_bool3(bool x)
{
    return bool3{x, x, x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool4 make_bool4(bool x)
{
    return bool4{x, x, x, x};
}

#if SLANG_CUDA_RTC
#define SLANG_MAKE_VECTOR_FROM_SCALAR(T)                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##1 make_##T##1(T x) \
    {                                                        \
        return T##1 {x};                                     \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x) \
    {                                                        \
        return make_##T##2(x, x);                            \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x) \
    {                                                        \
        return make_##T##3(x, x, x);                         \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x) \
    {                                                        \
        return make_##T##4(x, x, x, x);                      \
    }
#else
#define SLANG_MAKE_VECTOR_FROM_SCALAR(T)                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x) \
    {                                                        \
        return make_##T##2(x, x);                            \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x) \
    {                                                        \
        return make_##T##3(x, x, x);                         \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x) \
    {                                                        \
        return make_##T##4(x, x, x, x);                      \
    }
#endif
SLANG_MAKE_VECTOR_FROM_SCALAR(int)
SLANG_MAKE_VECTOR_FROM_SCALAR(uint)
SLANG_MAKE_VECTOR_FROM_SCALAR(short)
SLANG_MAKE_VECTOR_FROM_SCALAR(ushort)
SLANG_MAKE_VECTOR_FROM_SCALAR(char)
SLANG_MAKE_VECTOR_FROM_SCALAR(uchar)
SLANG_MAKE_VECTOR_FROM_SCALAR(longlong)
SLANG_MAKE_VECTOR_FROM_SCALAR(ulonglong)
SLANG_MAKE_VECTOR_FROM_SCALAR(float)
SLANG_MAKE_VECTOR_FROM_SCALAR(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_MAKE_VECTOR_FROM_SCALAR(__half)
#if !SLANG_CUDA_RTC
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half1 make___half1(__half x)
{
    return __half1{x};
}
#endif
#endif

#define SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(Fn, T, N)                                            \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##N Fn(T##N* address, T##N val)                           \
    {                                                                                             \
        T##N result;                                                                              \
        for (int i = 0; i < N; i++)                                                               \
            *_slang_vector_get_element_ptr(&result, i) =                                          \
                Fn(_slang_vector_get_element_ptr(address, i), _slang_vector_get_element(val, i)); \
        return result;                                                                            \
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 4)
#endif
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 4)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 4)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 4)

template<typename T, int n>
struct GetVectorTypeImpl
{
};

#define GET_VECTOR_TYPE_IMPL(T, n)                                     \
    template<>                                                         \
    struct GetVectorTypeImpl<T, n>                                     \
    {                                                                  \
        typedef T##n type;                                             \
        static SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n fromScalar(T v) \
        {                                                              \
            return make_##T##n(v);                                     \
        }                                                              \
    };
#define GET_VECTOR_TYPE_IMPL_N(T) \
    GET_VECTOR_TYPE_IMPL(T, 1)    \
    GET_VECTOR_TYPE_IMPL(T, 2)    \
    GET_VECTOR_TYPE_IMPL(T, 3)    \
    GET_VECTOR_TYPE_IMPL(T, 4)

GET_VECTOR_TYPE_IMPL_N(int)
GET_VECTOR_TYPE_IMPL_N(uint)
GET_VECTOR_TYPE_IMPL_N(short)
GET_VECTOR_TYPE_IMPL_N(ushort)
GET_VECTOR_TYPE_IMPL_N(char)
GET_VECTOR_TYPE_IMPL_N(uchar)
GET_VECTOR_TYPE_IMPL_N(longlong)
GET_VECTOR_TYPE_IMPL_N(ulonglong)
GET_VECTOR_TYPE_IMPL_N(float)
GET_VECTOR_TYPE_IMPL_N(double)
#if SLANG_CUDA_ENABLE_HALF
GET_VECTOR_TYPE_IMPL_N(__half)
#endif
template<typename T, int n>
using Vector = typename GetVectorTypeImpl<T, n>::type;

template<typename T, int n, typename OtherT, int m>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, n> _slang_vector_reshape(const Vector<OtherT, m> other)
{
    Vector<T, n> result;
    for (int i = 0; i < n; i++)
    {
        OtherT otherElement = T(0);
        if (i < m)
            otherElement = _slang_vector_get_element(other, i);
        *_slang_vector_get_element_ptr(&result, i) = (T)otherElement;
    }
    return result;
}

template<typename T, int ROWS, int COLS>
struct Matrix
{
    Vector<T, COLS> rows[ROWS];
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, COLS>& operator[](size_t index)
    {
        return rows[index];
    }
};


template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T scalar)
{
    Matrix<T, ROWS, COLS> result;
    for (int i = 0; i < ROWS; i++)
        result.rows[i] = GetVectorTypeImpl<T, COLS>::fromScalar(scalar);
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(const Vector<T, COLS>& row0)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1,
    const Vector<T, COLS>& row2)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    result.rows[2] = row2;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1,
    const Vector<T, COLS>& row2,
    const Vector<T, COLS>& row3)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    result.rows[2] = row2;
    result.rows[3] = row3;
    return result;
}

template<typename T, int ROWS, int COLS, typename U, int otherRow, int otherCol>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Matrix<U, otherRow, otherCol>& other)
{
    Matrix<T, ROWS, COLS> result;
    int minRow = ROWS;
    int minCol = COLS;
    if (minRow > otherRow)
        minRow = otherRow;
    if (minCol > otherCol)
        minCol = otherCol;
    for (int i = 0; i < minRow; i++)
        for (int j = 0; j < minCol; j++)
            *_slang_vector_get_element_ptr(result.rows + i, j) =
                (T)_slang_vector_get_element(other.rows[i], j);
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T v0, T v1, T v2, T v3)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[1].x = v2;
    rs.rows[1].y = v3;
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 3)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v5;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[1].x = v2;
        rs.rows[1].y = v3;
        rs.rows[2].x = v4;
        rs.rows[2].y = v5;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 4)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[0], 3) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v5;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v6;
        *_slang_vector_get_element_ptr(&rs.rows[1], 3) = v7;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[1].x = v2;
        rs.rows[1].y = v3;
        rs.rows[2].x = v4;
        rs.rows[2].y = v5;
        rs.rows[3].x = v6;
        rs.rows[3].y = v7;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[0].z = v2;
    rs.rows[1].x = v3;
    rs.rows[1].y = v4;
    rs.rows[1].z = v5;
    rs.rows[2].x = v6;
    rs.rows[2].y = v7;
    rs.rows[2].z = v8;
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8,
    T v9,
    T v10,
    T v11)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 4)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[0], 3) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v5;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v6;
        *_slang_vector_get_element_ptr(&rs.rows[1], 3) = v7;
        *_slang_vector_get_element_ptr(&rs.rows[2], 0) = v8;
        *_slang_vector_get_element_ptr(&rs.rows[2], 1) = v9;
        *_slang_vector_get_element_ptr(&rs.rows[2], 2) = v10;
        *_slang_vector_get_element_ptr(&rs.rows[2], 3) = v11;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[0].z = v2;
        rs.rows[1].x = v3;
        rs.rows[1].y = v4;
        rs.rows[1].z = v5;
        rs.rows[2].x = v6;
        rs.rows[2].y = v7;
        rs.rows[2].z = v8;
        rs.rows[3].x = v9;
        rs.rows[3].y = v10;
        rs.rows[3].z = v11;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8,
    T v9,
    T v10,
    T v11,
    T v12,
    T v13,
    T v14,
    T v15)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[0].z = v2;
    rs.rows[0].w = v3;
    rs.rows[1].x = v4;
    rs.rows[1].y = v5;
    rs.rows[1].z = v6;
    rs.rows[1].w = v7;
    rs.rows[2].x = v8;
    rs.rows[2].y = v9;
    rs.rows[2].z = v10;
    rs.rows[2].w = v11;
    rs.rows[3].x = v12;
    rs.rows[3].y = v13;
    rs.rows[3].z = v14;
    rs.rows[3].w = v15;
    return rs;
}

#define SLANG_MATRIX_BINARY_OP(T, op)                                   \
    template<int R, int C>                                              \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator op(     \
        const Matrix<T, R, C>& thisVal,                                 \
        const Matrix<T, R, C>& other)                                   \
    {                                                                   \
        Matrix<T, R, C> result;                                         \
        for (int i = 0; i < R; i++)                                     \
            for (int j = 0; j < C; j++)                                 \
                *_slang_vector_get_element_ptr(result.rows + i, j) =    \
                    _slang_vector_get_element(thisVal.rows[i], j)       \
                        op _slang_vector_get_element(other.rows[i], j); \
        return result;                                                  \
    }

#define SLANG_MATRIX_UNARY_OP(T, op)                                                               \
    template<int R, int C>                                                                         \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator op(const Matrix<T, R, C>& thisVal) \
    {                                                                                              \
        Matrix<T, R, C> result;                                                                    \
        for (int i = 0; i < R; i++)                                                                \
            for (int j = 0; j < C; j++)                                                            \
                *_slang_vector_get_element_ptr(result.rows + i, j) =                               \
                    op _slang_vector_get_element(thisVal.rows[i], j);                              \
        return result;                                                                             \
    }
#define SLANG_INT_MATRIX_OPS(T)   \
    SLANG_MATRIX_BINARY_OP(T, +)  \
    SLANG_MATRIX_BINARY_OP(T, -)  \
    SLANG_MATRIX_BINARY_OP(T, *)  \
    SLANG_MATRIX_BINARY_OP(T, /)  \
    SLANG_MATRIX_BINARY_OP(T, &)  \
    SLANG_MATRIX_BINARY_OP(T, |)  \
    SLANG_MATRIX_BINARY_OP(T, &&) \
    SLANG_MATRIX_BINARY_OP(T, ||) \
    SLANG_MATRIX_BINARY_OP(T, ^)  \
    SLANG_MATRIX_BINARY_OP(T, %)  \
    SLANG_MATRIX_UNARY_OP(T, !)   \
    SLANG_MATRIX_UNARY_OP(T, ~)
#define SLANG_FLOAT_MATRIX_OPS(T) \
    SLANG_MATRIX_BINARY_OP(T, +)  \
    SLANG_MATRIX_BINARY_OP(T, -)  \
    SLANG_MATRIX_BINARY_OP(T, *)  \
    SLANG_MATRIX_BINARY_OP(T, /)  \
    SLANG_MATRIX_UNARY_OP(T, -)
SLANG_INT_MATRIX_OPS(int)
SLANG_INT_MATRIX_OPS(uint)
SLANG_INT_MATRIX_OPS(short)
SLANG_INT_MATRIX_OPS(ushort)
SLANG_INT_MATRIX_OPS(char)
SLANG_INT_MATRIX_OPS(uchar)
SLANG_INT_MATRIX_OPS(longlong)
SLANG_INT_MATRIX_OPS(ulonglong)
SLANG_FLOAT_MATRIX_OPS(float)
SLANG_FLOAT_MATRIX_OPS(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_FLOAT_MATRIX_OPS(__half)
#endif
#define SLANG_MATRIX_INT_NEG_OP(T)                                                        \
    template<int R, int C>                                                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator-(Matrix<T, R, C> thisVal) \
    {                                                                                     \
        Matrix<T, R, C> result;                                                           \
        for (int i = 0; i < R; i++)                                                       \
            for (int j = 0; j < C; j++)                                                   \
                *_slang_vector_get_element_ptr(result.rows + i, j) =                      \
                    0 - _slang_vector_get_element(thisVal.rows[i], j);                    \
        return result;                                                                    \
    }
SLANG_MATRIX_INT_NEG_OP(int)
SLANG_MATRIX_INT_NEG_OP(uint)
SLANG_MATRIX_INT_NEG_OP(short)
SLANG_MATRIX_INT_NEG_OP(ushort)
SLANG_MATRIX_INT_NEG_OP(char)
SLANG_MATRIX_INT_NEG_OP(uchar)
SLANG_MATRIX_INT_NEG_OP(longlong)
SLANG_MATRIX_INT_NEG_OP(ulonglong)

#define SLANG_FLOAT_MATRIX_MOD(T)                                                 \
    template<int R, int C>                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator%(                 \
        Matrix<T, R, C> left,                                                     \
        Matrix<T, R, C> right)                                                    \
    {                                                                             \
        Matrix<T, R, C> result;                                                   \
        for (int i = 0; i < R; i++)                                               \
            for (int j = 0; j < C; j++)                                           \
                *_slang_vector_get_element_ptr(result.rows + i, j) = _slang_fmod( \
                    _slang_vector_get_element(left.rows[i], j),                   \
                    _slang_vector_get_element(right.rows[i], j));                 \
        return result;                                                            \
    }

SLANG_FLOAT_MATRIX_MOD(float)
SLANG_FLOAT_MATRIX_MOD(double)
#if SLANG_CUDA_ENABLE_HALF
template<int R, int C>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<__half, R, C> operator%(
    Matrix<__half, R, C> left,
    Matrix<__half, R, C> right)
{
    Matrix<__half, R, C> result;
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            *_slang_vector_get_element_ptr(result.rows + i, j) = __float2half(_slang_fmod(
                __half2float(_slang_vector_get_element(left.rows[i], j)),
                __half2float(_slang_vector_get_element(right.rows[i], j))));
    return result;
}
#endif
#undef SLANG_FLOAT_MATRIX_MOD
#undef SLANG_MATRIX_BINARY_OP
#undef SLANG_MATRIX_UNARY_OP
#undef SLANG_INT_MATRIX_OPS
#undef SLANG_FLOAT_MATRIX_OPS
#undef SLANG_MATRIX_INT_NEG_OP
#undef SLANG_FLOAT_MATRIX_MOD

#define SLANG_SELECT_IMPL(T, N)                                                                  \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, N> _slang_select(                               \
        bool##N condition,                                                                       \
        Vector<T, N> v0,                                                                         \
        Vector<T, N> v1)                                                                         \
    {                                                                                            \
        Vector<T, N> result;                                                                     \
        for (int i = 0; i < N; i++)                                                              \
        {                                                                                        \
            *_slang_vector_get_element_ptr(&result, i) = _slang_vector_get_element(condition, i) \
                                                             ? _slang_vector_get_element(v0, i)  \
                                                             : _slang_vector_get_element(v1, i); \
        }                                                                                        \
        return result;                                                                           \
    }
#define SLANG_SELECT_T(T)   \
    SLANG_SELECT_IMPL(T, 2) \
    SLANG_SELECT_IMPL(T, 3) \
    SLANG_SELECT_IMPL(T, 4)

SLANG_SELECT_T(int)
SLANG_SELECT_T(uint)
SLANG_SELECT_T(short)
SLANG_SELECT_T(ushort)
SLANG_SELECT_T(char)
SLANG_SELECT_T(uchar)
SLANG_SELECT_T(float)
SLANG_SELECT_T(double)

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_select(bool condition, T v0, T v1)
{
    return condition ? v0 : v1;
}

//
// Half support
//

#if SLANG_CUDA_ENABLE_HALF
SLANG_SELECT_T(__half)

// Convenience functions ushort -> half

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half2 __ushort_as_half(const ushort2& i)
{
    return __halves2half2(__ushort_as_half(i.x), __ushort_as_half(i.y));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half3 __ushort_as_half(const ushort3& i)
{
    return __half3{__ushort_as_half(i.x), __ushort_as_half(i.y), __ushort_as_half(i.z)};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half4 __ushort_as_half(const ushort4& i)
{
    return __half4{
        __ushort_as_half(i.x),
        __ushort_as_half(i.y),
        __ushort_as_half(i.z),
        __ushort_as_half(i.w)};
}

// Convenience functions half -> ushort

SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort2 __half_as_ushort(const __half2& i)
{
    return make_ushort2(__half_as_ushort(i.x), __half_as_ushort(i.y));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort3 __half_as_ushort(const __half3& i)
{
    return make_ushort3(__half_as_ushort(i.x), __half_as_ushort(i.y), __half_as_ushort(i.z));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort4 __half_as_ushort(const __half4& i)
{
    return make_ushort4(
        __half_as_ushort(i.x),
        __half_as_ushort(i.y),
        __half_as_ushort(i.z),
        __half_as_ushort(i.w));
}

// This is a little bit of a hack. Fortunately CUDA has the definitions of the templated types in
// include/surface_indirect_functions.h
// Here we find the template definition requires a specialization of __nv_isurf_trait to allow
// a specialization of the surface write functions.
// This *isn't* a problem on the read functions as they don't have a return type that uses this
// mechanism

template<>
struct __nv_isurf_trait<__half>
{
    typedef void type;
};
template<>
struct __nv_isurf_trait<__half2>
{
    typedef void type;
};
template<>
struct __nv_isurf_trait<__half4>
{
    typedef void type;
};

#define SLANG_DROP_PARENS(...) __VA_ARGS__

#define SLANG_SURFACE_READ(FUNC_NAME, TYPE_ARGS, ARGS)                                             \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half FUNC_NAME<__half>(                                   \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(FUNC_NAME<ushort>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half2 FUNC_NAME<__half2>(                                 \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(                                                                   \
            FUNC_NAME<ushort2>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                    \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half4 FUNC_NAME<__half4>(                                 \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(                                                                   \
            FUNC_NAME<ushort4>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                    \
    }

SLANG_SURFACE_READ(surf1Dread, (int x), (x))
SLANG_SURFACE_READ(surf2Dread, (int x, int y), (x, y))
SLANG_SURFACE_READ(surf3Dread, (int x, int y, int z), (x, y, z))
SLANG_SURFACE_READ(surf1DLayeredread, (int x, int layer), (x, layer))
SLANG_SURFACE_READ(surf2DLayeredread, (int x, int y, int layer), (x, y, layer))
SLANG_SURFACE_READ(surfCubemapread, (int x, int y, int face), (x, y, face))
SLANG_SURFACE_READ(surfCubemapLayeredread, (int x, int y, int layerFace), (x, y, layerFace))

#define SLANG_SURFACE_WRITE(FUNC_NAME, TYPE_ARGS, ARGS)                                            \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half>(                                     \
        __half data,                                                                               \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode);  \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half2>(                                    \
        __half2 data,                                                                              \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort2>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode); \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half4>(                                    \
        __half4 data,                                                                              \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort4>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode); \
    }

SLANG_SURFACE_WRITE(surf1Dwrite, (int x), (x))
SLANG_SURFACE_WRITE(surf2Dwrite, (int x, int y), (x, y))
SLANG_SURFACE_WRITE(surf3Dwrite, (int x, int y, int z), (x, y, z))
SLANG_SURFACE_WRITE(surf1DLayeredwrite, (int x, int layer), (x, layer))
SLANG_SURFACE_WRITE(surf2DLayeredwrite, (int x, int y, int layer), (x, y, layer))
SLANG_SURFACE_WRITE(surfCubemapwrite, (int x, int y, int face), (x, y, face))
SLANG_SURFACE_WRITE(surfCubemapLayeredwrite, (int x, int y, int layerFace), (x, y, layerFace))

// ! Hack to test out reading !!!
// Only works converting *from* half

// template <typename T>
// SLANG_FORCE_INLINE SLANG_CUDA_CALL T surf2Dread_convert(cudaSurfaceObject_t surfObj, int x, int
// y, cudaSurfaceBoundaryMode boundaryMode);

#define SLANG_SURFACE_READ_HALF_CONVERT(FUNC_NAME, TYPE_ARGS, ARGS)                              \
                                                                                                 \
    template<typename T>                                                                         \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T FUNC_NAME##_convert(                                    \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode);                                                   \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float FUNC_NAME##_convert<float>(                         \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        return __ushort_as_half(                                                                 \
            FUNC_NAME<uint16_t>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                 \
    }                                                                                            \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float2 FUNC_NAME##_convert<float2>(                       \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        const __half2 v =                                                                        \
            __ushort_as_half(FUNC_NAME<ushort2>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
        return float2{v.x, v.y};                                                                 \
    }                                                                                            \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float4 FUNC_NAME##_convert<float4>(                       \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        const __half4 v =                                                                        \
            __ushort_as_half(FUNC_NAME<ushort4>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
        return float4{v.x, v.y, v.z, v.w};                                                       \
    }

SLANG_SURFACE_READ_HALF_CONVERT(surf1Dread, (int x), (x))
SLANG_SURFACE_READ_HALF_CONVERT(surf2Dread, (int x, int y), (x, y))
SLANG_SURFACE_READ_HALF_CONVERT(surf3Dread, (int x, int y, int z), (x, y, z))

#endif

// Support for doing format conversion when writing to a surface/RWTexture

// NOTE! For normal surface access x values are *byte* addressed.
// For the _convert versions they are *not*. They don't need to be because sust.p does not require
// it.

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert(
    T,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode);
template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert(
    T,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode);
template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert(
    T,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode);

// https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-sust

// Float

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<float>(
    float v,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode)
{
    asm volatile(
        "{sust.p.1d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2};}\n\t" ::"l"(surfObj),
        "r"(x),
        "f"(v));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<float>(
    float v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode)
{
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2}], {%3};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "f"(v));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<float>(
    float v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode)
{
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2,%3}], {%4};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "r"(z),
        "f"(v));
}

// Float2

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<float2>(
    float2 v,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y;
    asm volatile(
        "{sust.p.1d.v2.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2,%3};}\n\t" ::"l"(surfObj),
        "r"(x),
        "f"(vx),
        "f"(vy));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<float2>(
    float2 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y;
    asm volatile(
        "{sust.p.2d.v2.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2}], {%3,%4};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "f"(vx),
        "f"(vy));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<float2>(
    float2 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y;
    asm volatile(
        "{sust.p.2d.v2.b32." SLANG_PTX_BOUNDARY_MODE
        " [%0, {%1,%2,%3}], {%4,%5};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "r"(z),
        "f"(vx),
        "f"(vy));
}

// Float4
template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<float4>(
    float4 v,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y, vz = v.z, vw = v.w;
    asm volatile(
        "{sust.p.1d.v4.b32." SLANG_PTX_BOUNDARY_MODE
        " [%0, {%1}], {%2,%3,%4,%5};}\n\t" ::"l"(surfObj),
        "r"(x),
        "f"(vx),
        "f"(vy),
        "f"(vz),
        "f"(vw));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<float4>(
    float4 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y, vz = v.z, vw = v.w;
    asm volatile(
        "{sust.p.2d.v4.b32." SLANG_PTX_BOUNDARY_MODE
        " [%0, {%1,%2}], {%3,%4,%5,%6};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "f"(vx),
        "f"(vy),
        "f"(vz),
        "f"(vw));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<float4>(
    float4 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y, vz = v.z, vw = v.w;
    asm volatile(
        "{sust.p.2d.v4.b32." SLANG_PTX_BOUNDARY_MODE
        " [%0, {%1,%2,%3}], {%4,%5,%6,%7};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "r"(z),
        "f"(vx),
        "f"(vy),
        "f"(vz),
        "f"(vw));
}

// ----------------------------- F32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_ceil(float f)
{
    return ::ceilf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_floor(float f)
{
    return ::floorf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_round(float f)
{
    return ::roundf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sin(float f)
{
    return ::sinf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_cos(float f)
{
    return ::cosf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL void F32_sincos(float f, float* s, float* c)
{
    ::sincosf(f, s, c);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_tan(float f)
{
    return ::tanf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_asin(float f)
{
    return ::asinf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_acos(float f)
{
    return ::acosf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_atan(float f)
{
    return ::atanf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sinh(float f)
{
    return ::sinhf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_cosh(float f)
{
    return ::coshf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_tanh(float f)
{
    return ::tanhf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log2(float f)
{
    return ::log2f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log(float f)
{
    return ::logf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log10(float f)
{
    return ::log10f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_exp2(float f)
{
    return ::exp2f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_exp(float f)
{
    return ::expf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_abs(float f)
{
    return ::fabsf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_trunc(float f)
{
    return ::truncf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sqrt(float f)
{
    return ::sqrtf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_rsqrt(float f)
{
    return ::rsqrtf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sign(float f)
{
    return (f == 0.0f) ? f : ((f < 0.0f) ? -1.0f : 1.0f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_frac(float f)
{
    return f - F32_floor(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isnan(float f)
{
    return isnan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isfinite(float f)
{
    return isfinite(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isinf(float f)
{
    return isinf(f);
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_min(float a, float b)
{
    return ::fminf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_max(float a, float b)
{
    return ::fmaxf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_pow(float a, float b)
{
    return ::powf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_fmod(float a, float b)
{
    return ::fmodf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_remainder(float a, float b)
{
    return ::remainderf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_atan2(float a, float b)
{
    return float(::atan2(a, b));
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_frexp(float x, int* e)
{
    return frexpf(x, e);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_modf(float x, float* ip)
{
    return ::modff(x, ip);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t F32_asuint(float f)
{
    Union32 u;
    u.f = f;
    return u.u;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t F32_asint(float f)
{
    Union32 u;
    u.f = f;
    return u.i;
}

// Ternary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_fma(float a, float b, float c)
{
    return ::fmaf(a, b, c);
}


// ----------------------------- F64 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_ceil(double f)
{
    return ::ceil(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_floor(double f)
{
    return ::floor(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_round(double f)
{
    return ::round(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sin(double f)
{
    return ::sin(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_cos(double f)
{
    return ::cos(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_sincos(double f, double* s, double* c)
{
    ::sincos(f, s, c);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_tan(double f)
{
    return ::tan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_asin(double f)
{
    return ::asin(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_acos(double f)
{
    return ::acos(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_atan(double f)
{
    return ::atan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sinh(double f)
{
    return ::sinh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_cosh(double f)
{
    return ::cosh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_tanh(double f)
{
    return ::tanh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log2(double f)
{
    return ::log2(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log(double f)
{
    return ::log(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log10(float f)
{
    return ::log10(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_exp2(double f)
{
    return ::exp2(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_exp(double f)
{
    return ::exp(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_abs(double f)
{
    return ::fabs(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_trunc(double f)
{
    return ::trunc(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sqrt(double f)
{
    return ::sqrt(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_rsqrt(double f)
{
    return ::rsqrt(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sign(double f)
{
    return (f == 0.0) ? f : ((f < 0.0) ? -1.0 : 1.0);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_frac(double f)
{
    return f - F64_floor(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isnan(double f)
{
    return isnan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isfinite(double f)
{
    return isfinite(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isinf(double f)
{
    return isinf(f);
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_min(double a, double b)
{
    return ::fmin(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_max(double a, double b)
{
    return ::fmax(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_pow(double a, double b)
{
    return ::pow(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_fmod(double a, double b)
{
    return ::fmod(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_remainder(double a, double b)
{
    return ::remainder(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_atan2(double a, double b)
{
    return ::atan2(a, b);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_frexp(double x, int* e)
{
    return ::frexp(x, e);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_modf(double x, double* ip)
{
    return ::modf(x, ip);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_asuint(double d, uint32_t* low, uint32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = uint32_t(u.u);
    *hi = uint32_t(u.u >> 32);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_asint(double d, int32_t* low, int32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = int32_t(u.u);
    *hi = int32_t(u.u >> 32);
}

// Ternary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_fma(double a, double b, double c)
{
    return ::fma(a, b, c);
}

// ----------------------------- U8 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U8_countbits(uint8_t v)
{
    // No native 8bit popc yet, just cast and use 32bit variant
    return __popc(uint32_t(v));
}

// ----------------------------- I8 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I8_countbits(int8_t v)
{
    return U8_countbits(uint8_t(v));
}

// ----------------------------- U16 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U16_countbits(uint16_t v)
{
    // No native 16bit popc yet, just cast and use 32bit variant
    return __popc(uint32_t(v));
}

// ----------------------------- I16 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I16_countbits(int16_t v)
{
    return U16_countbits(uint16_t(v));
}

// ----------------------------- U32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_abs(uint32_t f)
{
    return f;
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_min(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_max(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float U32_asfloat(uint32_t x)
{
    Union32 u;
    u.u = x;
    return u.f;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_asint(int32_t x)
{
    return uint32_t(x);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double U32_asdouble(uint32_t low, uint32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | low;
    return u.d;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_countbits(uint32_t v)
{
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html#group__CUDA__MATH__INTRINSIC__INT_1g43c9c7d2b9ebf202ff1ef5769989be46
    return __popc(v);
}

// ----------------------------- I32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_abs(int32_t f)
{
    return (f < 0) ? -f : f;
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_min(int32_t a, int32_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_max(int32_t a, int32_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float I32_asfloat(int32_t x)
{
    Union32 u;
    u.i = x;
    return u.f;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I32_asuint(int32_t x)
{
    return uint32_t(x);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double I32_asdouble(int32_t low, int32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | uint32_t(low);
    return u.d;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I32_countbits(int32_t v)
{
    return U32_countbits(uint32_t(v));
}

// ----------------------------- U64 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_abs(uint64_t f)
{
    return f;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_min(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_max(uint64_t a, uint64_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U64_countbits(uint64_t v)
{
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html#group__CUDA__MATH__INTRINSIC__INT_1g43c9c7d2b9ebf202ff1ef5769989be46
    return __popcll(v);
}

// ----------------------------- I64 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_abs(int64_t f)
{
    return (f < 0) ? -f : f;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_min(int64_t a, int64_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_max(int64_t a, int64_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I64_countbits(int64_t v)
{
    return U64_countbits(uint64_t(v));
}

// ----------------------------- IPTR -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL intptr_t IPTR_abs(intptr_t f)
{
    return (f < 0) ? -f : f;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL intptr_t IPTR_min(intptr_t a, intptr_t b)
{
    return a < b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL intptr_t IPTR_max(intptr_t a, intptr_t b)
{
    return a > b ? a : b;
}

// ----------------------------- UPTR -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL uintptr_t UPTR_abs(uintptr_t f)
{
    return f;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uintptr_t UPTR_min(uintptr_t a, uintptr_t b)
{
    return a < b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uintptr_t UPTR_max(uintptr_t a, uintptr_t b)
{
    return a > b ? a : b;
}

// ----------------------------- ResourceType -----------------------------------------


// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-structuredbuffer-getdimensions
// Missing  Load(_In_  int  Location, _Out_ uint Status);

template<typename T>
struct StructuredBuffer
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, count);
#endif
        return data[index];
    }

    SLANG_CUDA_CALL const T& Load(size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, count);
#endif
        return data[index];
    }

#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outNumStructs, uint32_t* outStride)
    {
        *outNumStructs = uint32_t(count);
        *outStride = uint32_t(sizeof(T));
    }
#endif

    T* data;
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
    size_t count;
#endif
};

template<typename T>
struct RWStructuredBuffer : StructuredBuffer<T>
{
    SLANG_CUDA_CALL T& operator[](size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, this->count);
#endif
        return this->data[index];
    }
};

// Missing  Load(_In_  int  Location, _Out_ uint Status);
struct ByteAddressBuffer
{
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }
    SLANG_CUDA_CALL uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    SLANG_CUDA_CALL uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    SLANG_CUDA_CALL uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    SLANG_CUDA_CALL uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    SLANG_CUDA_CALL T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        T data;
        memcpy(&data, ((const char*)this->data) + index, sizeof(T));
        return data;
    }
    template<typename T>
    SLANG_CUDA_CALL StructuredBuffer<T> asStructuredBuffer() const
    {
        StructuredBuffer<T> rs;
        rs.data = (T*)data;
        rs.count = sizeInBytes / sizeof(T);
        return rs;
    }
    const uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};

// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-rwbyteaddressbuffer
// Missing support for Atomic operations
// Missing support for Load with status
struct RWByteAddressBuffer
{
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }

    SLANG_CUDA_CALL uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    SLANG_CUDA_CALL uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    SLANG_CUDA_CALL uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    SLANG_CUDA_CALL uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    SLANG_CUDA_CALL T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        T data;
        memcpy(&data, ((const char*)this->data) + index, sizeof(T));
        return data;
    }

    SLANG_CUDA_CALL void Store(size_t index, uint32_t v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        data[index >> 2] = v;
    }
    SLANG_CUDA_CALL void Store2(size_t index, uint2 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
    }
    SLANG_CUDA_CALL void Store3(size_t index, uint3 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
    }
    SLANG_CUDA_CALL void Store4(size_t index, uint4 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
        data[dataIdx + 3] = v.w;
    }
    template<typename T>
    SLANG_CUDA_CALL void Store(size_t index, T const& value) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        memcpy((char*)data + index, &value, sizeof(T));
    }

    /// Can be used in the core module to gain access
    template<typename T>
    SLANG_CUDA_CALL T* _getPtrAt(size_t index)
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        return (T*)(((char*)data) + index);
    }
    template<typename T>
    SLANG_CUDA_CALL RWStructuredBuffer<T> asStructuredBuffer() const
    {
        RWStructuredBuffer<T> rs;
        rs.data = (T*)data;
        rs.count = sizeInBytes / sizeof(T);
        return rs;
    }
    uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};


// ---------------------- Wave --------------------------------------

// TODO(JS): It appears that cuda does not have a simple way to get a lane index.
//
// Another approach could be...
// laneId = ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) &
// SLANG_CUDA_WARP_MASK If that is really true another way to do this, would be for code generator
// to add this function with the [numthreads] baked in.
//
// For now I'll just assume you have a launch that makes the following correct if the kernel uses
// WaveGetLaneIndex()
#ifndef SLANG_USE_ASM_LANE_ID
__forceinline__ __device__ uint32_t _getLaneId()
{
    // If the launch is (or I guess some multiple of the warp size)
    // we try this mechanism, which is apparently faster.
    return threadIdx.x & SLANG_CUDA_WARP_MASK;
}
#else
__forceinline__ __device__ uint32_t _getLaneId()
{
    // https://stackoverflow.com/questions/44337309/whats-the-most-efficient-way-to-calculate-the-warp-id-lane-id-in-a-1-d-grid#
    // This mechanism is not the fastest way to do it, and that is why the other mechanism
    // is the default. But the other mechanism relies on a launch that makes the assumption
    // true.
    unsigned ret;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}
#endif

typedef int WarpMask;

// It appears that the __activemask() cannot always be used because
// threads need to be converged.
//
// For CUDA the article claims mask has to be used carefully
// https://devblogs.nvidia.com/using-cuda-warp-level-primitives/
// With the Warp intrinsics there is no mask, and it's just the 'active lanes'.
// __activemask() though does not require there is convergence, so that doesn't work.
//
// '__ballot_sync' produces a convergance.
//
// From the CUDA docs:
// ```For __all_sync, __any_sync, and __ballot_sync, a mask must be passed that specifies the
// threads participating in the call. A bit, representing the thread's lane ID, must be set for each
// participating thread to ensure they are properly converged before the intrinsic is executed by
// the hardware. All active threads named in mask must execute the same intrinsic with the same
// mask, or the result is undefined.```
//
// Currently there isn't a mechanism to correctly get the mask without it being passed through.
// Doing so will most likely require some changes to slang code generation to track masks, for now
// then we use _getActiveMask.

// Return mask of all the lanes less than the current lane
__forceinline__ __device__ WarpMask _getLaneLtMask()
{
    return (int(1) << _getLaneId()) - 1;
}

// TODO(JS):
// THIS IS NOT CORRECT! That determining the appropriate active mask requires appropriate
// mask tracking.
__forceinline__ __device__ WarpMask _getActiveMask()
{
    return __ballot_sync(__activemask(), true);
}

// Return a mask suitable for the 'MultiPrefix' style functions
__forceinline__ __device__ WarpMask _getMultiPrefixMask(int mask)
{
    return mask;
}

// Note! Note will return true if mask is 0, but thats okay, because there must be one
// lane active to execute anything
__inline__ __device__ bool _waveIsSingleLane(WarpMask mask)
{
    return (mask & (mask - 1)) == 0;
}

// Returns the power of 2 size of run of set bits. Returns 0 if not a suitable run.
// Examples:
// 0b00000000'00000000'00000000'11111111 -> 8
// 0b11111111'11111111'11111111'11111111 -> 32
// 0b00000000'00000000'00000000'00011111 -> 0 (since 5 is not a power of 2)
// 0b00000000'00000000'00000000'11110000 -> 0 (since the run of bits does not start at the LSB)
// 0b00000000'00000000'00000000'00100111 -> 0 (since it is not a single contiguous run)
__inline__ __device__ int _waveCalcPow2Offset(WarpMask mask)
{
    // This should be the most common case, so fast path it
    if (mask == SLANG_CUDA_WARP_BITMASK)
    {
        return SLANG_CUDA_WARP_SIZE;
    }
    // Is it a contiguous run of bits?
    if ((mask & (mask + 1)) == 0)
    {
        // const int offsetSize = __ffs(mask + 1) - 1;
        const int offset = 32 - __clz(mask);
        // Is it a power of 2 size
        if ((offset & (offset - 1)) == 0)
        {
            return offset;
        }
    }
    return 0;
}

__inline__ __device__ bool _waveIsFirstLane()
{
    const WarpMask mask = __activemask();
    // We special case bit 0, as that most warps are expected to be fully active.

    // mask & -mask, isolates the lowest set bit.
    // return (mask & 1 ) || ((mask & -mask) == (1 << _getLaneId()));

    // This mechanism is most similar to what was in an nVidia post, so assume it is prefered.
    return (mask & 1) || ((__ffs(mask) - 1) == _getLaneId());
}

template<typename T>
struct WaveOpOr
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a | b; }
};

template<typename T>
struct WaveOpAnd
{
    __inline__ __device__ static T getInitial(T a) { return ~T(0); }
    __inline__ __device__ static T doOp(T a, T b) { return a & b; }
};

template<typename T>
struct WaveOpXor
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a ^ b; }
    __inline__ __device__ static T doInverse(T a, T b) { return a ^ b; }
};

template<typename T>
struct WaveOpAdd
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a + b; }
    __inline__ __device__ static T doInverse(T a, T b) { return a - b; }
};

template<typename T>
struct WaveOpMul
{
    __inline__ __device__ static T getInitial(T a) { return T(1); }
    __inline__ __device__ static T doOp(T a, T b) { return a * b; }
    // Using this inverse for int is probably undesirable - because in general it requires T to have
    // more precision There is also a performance aspect to it, where divides are generally
    // significantly slower
    __inline__ __device__ static T doInverse(T a, T b) { return a / b; }
};

template<typename T>
struct WaveOpMax
{
    __inline__ __device__ static T getInitial(T a) { return a; }
    __inline__ __device__ static T doOp(T a, T b) { return a > b ? a : b; }
};

template<typename T>
struct WaveOpMin
{
    __inline__ __device__ static T getInitial(T a) { return a; }
    __inline__ __device__ static T doOp(T a, T b) { return a < b ? a : b; }
};

template<typename T>
struct ElementTypeTrait;

// Scalar
template<>
struct ElementTypeTrait<int>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<uint>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<float>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<double>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<uint64_t>
{
    typedef uint64_t Type;
};
template<>
struct ElementTypeTrait<int64_t>
{
    typedef int64_t Type;
};

// Vector
template<>
struct ElementTypeTrait<int1>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int2>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int3>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int4>
{
    typedef int Type;
};

template<>
struct ElementTypeTrait<uint1>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint2>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint3>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint4>
{
    typedef uint Type;
};

template<>
struct ElementTypeTrait<float1>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float2>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float3>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float4>
{
    typedef float Type;
};

template<>
struct ElementTypeTrait<double1>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double2>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double3>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double4>
{
    typedef double Type;
};

// Matrix
template<typename T, int ROWS, int COLS>
struct ElementTypeTrait<Matrix<T, ROWS, COLS>>
{
    typedef T Type;
};

// Scalar
template<typename INTF, typename T>
__device__ T _waveReduceScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);
    if (offsetSize > 0)
    {
        // Fast path O(log2(activeLanes))
        for (int offset = offsetSize >> 1; offset > 0; offset >>= 1)
        {
            val = INTF::doOp(val, __shfl_xor_sync(mask, val, offset));
        }
    }
    else if (!_waveIsSingleLane(mask))
    {
        T result = INTF::getInitial(val);
        int remaining = mask;
        while (remaining)
        {
            const int laneBit = remaining & -remaining;
            // Get the sourceLane
            const int srcLane = __ffs(laneBit) - 1;
            // Broadcast (can also broadcast to self)
            result = INTF::doOp(result, __shfl_sync(mask, val, srcLane));
            remaining &= ~laneBit;
        }
        return result;
    }
    return val;
}


// Multiple values
template<typename INTF, typename T, size_t COUNT>
__device__ void _waveReduceMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);
    if (offsetSize > 0)
    {
        // Fast path O(log2(activeLanes))
        for (int offset = offsetSize >> 1; offset > 0; offset >>= 1)
        {
            for (size_t i = 0; i < COUNT; ++i)
            {
                val[i] = INTF::doOp(val[i], __shfl_xor_sync(mask, val[i], offset));
            }
        }
    }
    else if (!_waveIsSingleLane(mask))
    {
        // Copy the original
        T originalVal[COUNT];
        for (size_t i = 0; i < COUNT; ++i)
        {
            const T v = val[i];
            originalVal[i] = v;
            val[i] = INTF::getInitial(v);
        }

        int remaining = mask;
        while (remaining)
        {
            const int laneBit = remaining & -remaining;
            // Get the sourceLane
            const int srcLane = __ffs(laneBit) - 1;
            // Broadcast (can also broadcast to self)
            for (size_t i = 0; i < COUNT; ++i)
            {
                val[i] = INTF::doOp(val[i], __shfl_sync(mask, originalVal[i], srcLane));
            }
            remaining &= ~laneBit;
        }
    }
}

template<typename INTF, typename T>
__device__ void _waveReduceMultiple(WarpMask mask, T* val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<INTF, ElemType, sizeof(T) / sizeof(ElemType)>(mask, (ElemType*)val);
}

template<typename T>
__inline__ __device__ T _waveOr(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpOr<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveAnd(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpAnd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveXor(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpXor<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveProduct(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMul<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveSum(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpAdd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveMin(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMin<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveMax(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMax<T>, T>(mask, val);
}

// Fast-path specializations when CUDA warp reduce operators are available
#if __CUDA_ARCH__ >= 800 // 8.x or higher
template<>
__inline__ __device__ unsigned _waveOr<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_or_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveAnd<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_and_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveXor<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_xor_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveSum<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_add_sync(mask, val);
}

template<>
__inline__ __device__ int _waveSum<int>(WarpMask mask, int val)
{
    return __reduce_add_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveMin<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_min_sync(mask, val);
}

template<>
__inline__ __device__ int _waveMin<int>(WarpMask mask, int val)
{
    return __reduce_min_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveMax<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_max_sync(mask, val);
}

template<>
__inline__ __device__ int _waveMax<int>(WarpMask mask, int val)
{
    return __reduce_max_sync(mask, val);
}
#endif


// Multiple

template<typename T>
__inline__ __device__ T _waveOrMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpOr<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveAndMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpAnd<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveXorMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpXor<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveProductMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMul<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveSumMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpAdd<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveMinMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMin<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveMaxMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMax<ElemType>>(mask, &val);
    return val;
}


template<typename T>
__inline__ __device__ bool _waveAllEqual(WarpMask mask, T val)
{
    int pred;
    __match_all_sync(mask, val, &pred);
    return pred != 0;
}

template<typename T>
__inline__ __device__ bool _waveAllEqualMultiple(WarpMask mask, T inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    int pred;
    const ElemType* src = (const ElemType*)&inVal;
    for (size_t i = 0; i < count; ++i)
    {
        __match_all_sync(mask, src[i], &pred);
        if (pred == 0)
        {
            return false;
        }
    }
    return true;
}

template<typename T>
__inline__ __device__ T _waveReadFirst(WarpMask mask, T val)
{
    const int lowestLaneId = __ffs(mask) - 1;
    return __shfl_sync(mask, val, lowestLaneId);
}

template<typename T>
__inline__ __device__ T _waveReadFirstMultiple(WarpMask mask, T inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    T outVal;
    const ElemType* src = (const ElemType*)&inVal;
    ElemType* dst = (ElemType*)&outVal;
    const int lowestLaneId = __ffs(mask) - 1;
    for (size_t i = 0; i < count; ++i)
    {
        dst[i] = __shfl_sync(mask, src[i], lowestLaneId);
    }
    return outVal;
}

template<typename T>
__inline__ __device__ T _waveShuffleMultiple(WarpMask mask, T inVal, int lane)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    T outVal;
    const ElemType* src = (const ElemType*)&inVal;
    ElemType* dst = (ElemType*)&outVal;
    for (size_t i = 0; i < count; ++i)
    {
        dst[i] = __shfl_sync(mask, src[i], lane);
    }
    return outVal;
}

// Scalar

// Invertable means that when we get to the end of the reduce, we can remove val (to make
// exclusive), using the inverse of the op.
template<typename INTF, typename T>
__device__ T _wavePrefixInvertableScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T result;
    if (offsetSize > 0)
    {
        // Sum is calculated inclusive of this lanes value
        result = val;
        for (int i = 1; i < offsetSize; i += i)
        {
            const T readVal = __shfl_up_sync(mask, result, i, offsetSize);
            if (laneId >= i)
            {
                result = INTF::doOp(result, readVal);
            }
        }
        // Remove val from the result, by applyin inverse
        result = INTF::doInverse(result, val);
    }
    else
    {
        result = INTF::getInitial(val);
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;
                // Broadcast (can also broadcast to self)
                const T readValue = __shfl_sync(mask, val, srcLane);
                // Only accumulate if srcLane is less than this lane
                if (srcLane < laneId)
                {
                    result = INTF::doOp(result, readValue);
                }
                remaining &= ~laneBit;
            }
        }
    }
    return result;
}


// This implementation separately tracks the value to be propogated, and the value
// that is the final result
template<typename INTF, typename T>
__device__ T _wavePrefixScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T result = INTF::getInitial(val);
    if (offsetSize > 0)
    {
        // For transmitted value we will do it inclusively with this lanes value
        // For the result we do not include the lanes value. This means an extra multiply for each
        // iteration but means we don't need to have a divide at the end and also removes overflow
        // issues in that scenario.
        for (int i = 1; i < offsetSize; i += i)
        {
            const T readVal = __shfl_up_sync(mask, val, i, offsetSize);
            if (laneId >= i)
            {
                result = INTF::doOp(result, readVal);
                val = INTF::doOp(val, readVal);
            }
        }
    }
    else
    {
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;
                // Broadcast (can also broadcast to self)
                const T readValue = __shfl_sync(mask, val, srcLane);
                // Only accumulate if srcLane is less than this lane
                if (srcLane < laneId)
                {
                    result = INTF::doOp(result, readValue);
                }
                remaining &= ~laneBit;
            }
        }
    }
    return result;
}


template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpCopy(T* dst, const T* src)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        dst[j] = src[j];
    }
}


template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpDoInverse(T* inOut, const T* val)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        inOut[j] = INTF::doInverse(inOut[j], val[j]);
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpSetInitial(T* out, const T* val)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        out[j] = INTF::getInitial(val[j]);
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _wavePrefixInvertableMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T originalVal[COUNT];
    _waveOpCopy<INTF, T, COUNT>(originalVal, val);

    if (offsetSize > 0)
    {
        // Sum is calculated inclusive of this lanes value
        for (int i = 1; i < offsetSize; i += i)
        {
            // TODO(JS): Note that here I don't split the laneId outside so it's only tested once.
            // This may be better but it would also mean that there would be shfl between lanes
            // that are on different (albeit identical) instructions. So this seems more likely to
            // work as expected with everything in lock step.
            for (size_t j = 0; j < COUNT; ++j)
            {
                const T readVal = __shfl_up_sync(mask, val[j], i, offsetSize);
                if (laneId >= i)
                {
                    val[j] = INTF::doOp(val[j], readVal);
                }
            }
        }
        // Remove originalVal from the result, by applyin inverse
        _waveOpDoInverse<INTF, T, COUNT>(val, originalVal);
    }
    else
    {
        _waveOpSetInitial<INTF, T, COUNT>(val, val);
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;

                for (size_t j = 0; j < COUNT; ++j)
                {
                    // Broadcast (can also broadcast to self)
                    const T readValue = __shfl_sync(mask, originalVal[j], srcLane);
                    // Only accumulate if srcLane is less than this lane
                    if (srcLane < laneId)
                    {
                        val[j] = INTF::doOp(val[j], readValue);
                    }
                    remaining &= ~laneBit;
                }
            }
        }
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _wavePrefixMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();

    T work[COUNT];
    _waveOpCopy<INTF, T, COUNT>(work, val);
    _waveOpSetInitial<INTF, T, COUNT>(val, val);

    if (offsetSize > 0)
    {
        // For transmitted value we will do it inclusively with this lanes value
        // For the result we do not include the lanes value. This means an extra op for each
        // iteration but means we don't need to have a divide at the end and also removes overflow
        // issues in that scenario.
        for (int i = 1; i < offsetSize; i += i)
        {
            for (size_t j = 0; j < COUNT; ++j)
            {
                const T readVal = __shfl_up_sync(mask, work[j], i, offsetSize);
                if (laneId >= i)
                {
                    work[j] = INTF::doOp(work[j], readVal);
                    val[j] = INTF::doOp(val[j], readVal);
                }
            }
        }
    }
    else
    {
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;

                for (size_t j = 0; j < COUNT; ++j)
                {
                    // Broadcast (can also broadcast to self)
                    const T readValue = __shfl_sync(mask, work[j], srcLane);
                    // Only accumulate if srcLane is less than this lane
                    if (srcLane < laneId)
                    {
                        val[j] = INTF::doOp(val[j], readValue);
                    }
                }
                remaining &= ~laneBit;
            }
        }
    }
}

template<typename T>
__inline__ __device__ T _wavePrefixProduct(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpMul<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixSum(WarpMask mask, T val)
{
    return _wavePrefixInvertableScalar<WaveOpAdd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixXor(WarpMask mask, T val)
{
    return _wavePrefixInvertableScalar<WaveOpXor<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixOr(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpOr<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixAnd(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpAnd<T>, T>(mask, val);
}


template<typename T>
__inline__ __device__ T _wavePrefixProductMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpMul<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixSumMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpAdd<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixXorMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpXor<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixOrMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpOr<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixAndMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpAnd<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ uint4 _waveMatchScalar(WarpMask mask, T val)
{
    int pred;
    return make_uint4(__match_all_sync(mask, val, &pred), 0, 0, 0);
}

template<typename T>
__inline__ __device__ uint4 _waveMatchMultiple(WarpMask mask, const T& inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    int pred;
    const ElemType* src = (const ElemType*)&inVal;
    uint matchBits = 0xffffffff;
    for (size_t i = 0; i < count && matchBits; ++i)
    {
        matchBits = matchBits & __match_all_sync(mask, src[i], &pred);
    }
    return make_uint4(matchBits, 0, 0, 0);
}

__device__ uint getAt(dim3 a, int b)
{
    SLANG_PRELUDE_ASSERT(b >= 0 && b < 3);
    return (&a.x)[b];
}
__device__ uint3 operator*(uint3 a, dim3 b)
{
    uint3 r;
    r.x = a.x * b.x;
    r.y = a.y * b.y;
    r.z = a.z * b.z;
    return r;
}

template<typename TResult, typename TInput>
__inline__ __device__ TResult slang_bit_cast(TInput val)
{
    return *(TResult*)(&val);
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */


/* Type that defines the uniform entry point params. The actual content of this type is dependent on
the entry point parameters, and can be found via reflection or defined such that it matches the
shader appropriately.
*/
struct UniformEntryPointParams;
struct UniformState;

// ---------------------- OptiX Ray Payload --------------------------------------
#ifdef SLANG_CUDA_ENABLE_OPTIX
struct RayDesc
{
    float3 Origin;
    float TMin;
    float3 Direction;
    float TMax;
};

static __forceinline__ __device__ void* unpackOptiXRayPayloadPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__ void packOptiXRayPayloadPointer(
    void* ptr,
    uint32_t& i0,
    uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void* getOptiXRayPayloadPtr()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return unpackOptiXRayPayloadPointer(u0, u1);
}

template<typename T>
__forceinline__ __device__ void* optixTrace(
    OptixTraversableHandle AccelerationStructure,
    uint32_t RayFlags,
    uint32_t InstanceInclusionMask,
    uint32_t RayContributionToHitGroupIndex,
    uint32_t MultiplierForGeometryContributionToHitGroupIndex,
    uint32_t MissShaderIndex,
    RayDesc Ray,
    T* Payload)
{
    uint32_t r0, r1;
    packOptiXRayPayloadPointer((void*)Payload, r0, r1);
    optixTrace(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        Ray.TMax,
        0.f, /* Time for motion blur, currently unsupported in slang */
        InstanceInclusionMask,
        RayFlags,
        RayContributionToHitGroupIndex,
        MultiplierForGeometryContributionToHitGroupIndex,
        MissShaderIndex,
        r0,
        r1);
}

__forceinline__ __device__ float4 optixGetSpherePositionAndRadius()
{
    float4 data[1];
    optixGetSphereData(data);
    return data[0];
}

__forceinline__ __device__ float4
optixHitObjectGetSpherePositionAndRadius(OptixTraversableHandle* Obj)
{
    float4 data[1];
    optixHitObjectGetSphereData(data);
    return data[0];
}

__forceinline__ __device__ Matrix<float, 2, 4> optixGetLssPositionsAndRadii()
{
    float4 data[2];
    optixGetLinearCurveVertexData(data);
    return makeMatrix<float, 2, 4>(data[0], data[1]);
}

__forceinline__ __device__ Matrix<float, 2, 4> optixHitObjectGetLssPositionsAndRadii(
    OptixTraversableHandle* Obj)
{
    float4 data[2];
    optixHitObjectGetLinearCurveVertexData(data);
    return makeMatrix<float, 2, 4>(data[0], data[1]);
}

__forceinline__ __device__ bool optixIsSphereHit()
{
    return optixGetPrimitiveType() == OPTIX_PRIMITIVE_TYPE_SPHERE;
}

__forceinline__ __device__ bool optixHitObjectIsSphereHit(OptixTraversableHandle* Obj)
{
    return optixGetPrimitiveType(optixHitObjectGetHitKind()) == OPTIX_PRIMITIVE_TYPE_SPHERE;
}

__forceinline__ __device__ bool optixIsLSSHit()
{
    return optixGetPrimitiveType() == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
}

__forceinline__ __device__ bool optixHitObjectIsLSSHit(OptixTraversableHandle* Obj)
{
    return optixGetPrimitiveType(optixHitObjectGetHitKind()) == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
}

template<typename T>
__forceinline__ __device__ void* optixTraverse(
    OptixTraversableHandle AccelerationStructure,
    uint32_t RayFlags,
    uint32_t InstanceInclusionMask,
    uint32_t RayContributionToHitGroupIndex,
    uint32_t MultiplierForGeometryContributionToHitGroupIndex,
    uint32_t MissShaderIndex,
    RayDesc Ray,
    T* Payload,
    OptixTraversableHandle* hitObj)
{
    uint32_t r0, r1;
    packOptiXRayPayloadPointer((void*)Payload, r0, r1);
    optixTraverse(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        Ray.TMax,
        0.f, /* Time for motion blur, currently unsupported in slang */
        InstanceInclusionMask,
        RayFlags,
        RayContributionToHitGroupIndex,
        MultiplierForGeometryContributionToHitGroupIndex,
        MissShaderIndex,
        r0,
        r1);
}

template<typename T>
__forceinline__ __device__ void* optixTraverse(
    OptixTraversableHandle AccelerationStructure,
    uint32_t RayFlags,
    uint32_t InstanceInclusionMask,
    uint32_t RayContributionToHitGroupIndex,
    uint32_t MultiplierForGeometryContributionToHitGroupIndex,
    uint32_t MissShaderIndex,
    RayDesc Ray,
    float RayTime,
    T* Payload,
    OptixTraversableHandle* hitObj)
{
    uint32_t r0, r1;
    packOptiXRayPayloadPointer((void*)Payload, r0, r1);
    optixTraverse(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        Ray.TMax,
        RayTime,
        InstanceInclusionMask,
        RayFlags,
        RayContributionToHitGroupIndex,
        MultiplierForGeometryContributionToHitGroupIndex,
        MissShaderIndex,
        r0,
        r1);
}

static __forceinline__ __device__ bool optixHitObjectIsHit(OptixTraversableHandle* hitObj)
{
    return optixHitObjectIsHit();
}

static __forceinline__ __device__ bool optixHitObjectIsMiss(OptixTraversableHandle* hitObj)
{
    return optixHitObjectIsMiss();
}

static __forceinline__ __device__ bool optixHitObjectIsNop(OptixTraversableHandle* hitObj)
{
    return optixHitObjectIsNop();
}

static __forceinline__ __device__ uint optixHitObjectGetClusterId(OptixTraversableHandle* hitObj)
{
    return optixHitObjectGetClusterId();
}

static __forceinline__ __device__ void optixMakeMissHitObject(
    uint MissShaderIndex,
    RayDesc Ray,
    OptixTraversableHandle* missObj)
{

    optixMakeMissHitObject(
        MissShaderIndex,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        Ray.TMax,
        0.f, /* rayTime */
        OPTIX_RAY_FLAG_NONE /* rayFlags*/);
}

static __forceinline__ __device__ void optixMakeMissHitObject(
    uint MissShaderIndex,
    RayDesc Ray,
    float CurrentTime,
    OptixTraversableHandle* missObj)
{

    optixMakeMissHitObject(
        MissShaderIndex,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        Ray.TMax,
        CurrentTime,
        OPTIX_RAY_FLAG_NONE /* rayFlags*/);
}

template<typename T>
static __forceinline__ __device__ void optixMakeHitObject(
    OptixTraversableHandle AccelerationStructure,
    uint InstanceIndex,
    uint GeometryIndex,
    uint PrimitiveIndex,
    uint HitKind,
    uint RayContributionToHitGroupIndex,
    uint MultiplierForGeometryContributionToHitGroupIndex,
    RayDesc Ray,
    T attr,
    OptixTraversableHandle* handle)
{

    OptixTraverseData data{};
    optixHitObjectGetTraverseData(&data);
    optixMakeHitObject(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        0.f,
        OPTIX_RAY_FLAG_NONE, /* rayFlags*/
        data,
        nullptr, /*OptixTraversableHandle* transforms*/
        0 /*numTransforms */);
}

template<typename T>
static __forceinline__ __device__ void optixMakeHitObject(
    uint HitGroupRecordIndex,
    OptixTraversableHandle AccelerationStructure,
    uint InstanceIndex,
    uint GeometryIndex,
    uint PrimitiveIndex,
    uint HitKind,
    RayDesc Ray,
    T attr,
    OptixTraversableHandle* handle)
{

    OptixTraverseData data{};
    optixHitObjectGetTraverseData(&data);
    optixMakeHitObject(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        0.f,
        OPTIX_RAY_FLAG_NONE, /* rayFlags*/
        data,
        nullptr, /*OptixTraversableHandle* transforms*/
        0 /*numTransforms */);
}

template<typename T>
static __forceinline__ __device__ void optixMakeHitObject(
    OptixTraversableHandle AccelerationStructure,
    uint InstanceIndex,
    uint GeometryIndex,
    uint PrimitiveIndex,
    uint HitKind,
    uint RayContributionToHitGroupIndex,
    uint MultiplierForGeometryContributionToHitGroupIndex,
    RayDesc Ray,
    float CurrentTime,
    T attr,
    OptixTraversableHandle* handle)
{

    OptixTraverseData data{};
    optixHitObjectGetTraverseData(&data);
    optixMakeHitObject(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        CurrentTime,
        OPTIX_RAY_FLAG_NONE, /* rayFlags*/
        data,
        nullptr, /*OptixTraversableHandle* transforms*/
        0 /*numTransforms */);
}

template<typename T>
static __forceinline__ __device__ void optixMakeHitObject(
    uint HitGroupRecordIndex,
    OptixTraversableHandle AccelerationStructure,
    uint InstanceIndex,
    uint GeometryIndex,
    uint PrimitiveIndex,
    uint HitKind,
    RayDesc Ray,
    float CurrentTime,
    T attr,
    OptixTraversableHandle* handle)
{

    OptixTraverseData data{};
    optixHitObjectGetTraverseData(&data);
    optixMakeHitObject(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        CurrentTime,
        OPTIX_RAY_FLAG_NONE, /* rayFlags*/
        data,
        nullptr, /*OptixTraversableHandle* transforms*/
        0 /*numTransforms */);
}

static __forceinline__ __device__ void optixMakeNopHitObject(OptixTraversableHandle* Obj)
{
    optixMakeNopHitObject();
}

template<typename T>
static __forceinline__ __device__ void optixInvoke(
    OptixTraversableHandle AccelerationStructure,
    OptixTraversableHandle* HitOrMiss,
    T Payload)
{
    uint32_t r0, r1;
    packOptiXRayPayloadPointer((void*)Payload, r0, r1);
    optixInvoke(r0, r1);
}
static __forceinline__ __device__ RayDesc optixHitObjectGetRayDesc(OptixTraversableHandle* obj)
{
    RayDesc ray = {
        optixHitObjectGetWorldRayOrigin(),
        optixHitObjectGetRayTmin(),
        optixHitObjectGetWorldRayDirection(),
        optixHitObjectGetRayTmax()};
    return ray;
}

static __forceinline__ __device__ uint optixHitObjectGetInstanceIndex(OptixTraversableHandle* Obj)
{
    return optixHitObjectGetInstanceIndex();
}

static __forceinline__ __device__ uint optixHitObjectGetInstanceId(OptixTraversableHandle* Obj)
{
    return optixHitObjectGetInstanceId();
}

static __forceinline__ __device__ uint optixHitObjectGetSbtGASIndex(OptixTraversableHandle* Obj)
{
    return optixHitObjectGetSbtGASIndex();
}

static __forceinline__ __device__ uint optixHitObjectGetPrimitiveIndex(OptixTraversableHandle* Obj)
{
    return optixHitObjectGetPrimitiveIndex();
}

template<typename T>
static __forceinline__ __device__ T optixHitObjectGetAttribute(OptixTraversableHandle* Obj)
{
    constexpr size_t numInts = (sizeof(T) + sizeof(uint32_t) - 1) /
                               sizeof(uint32_t); // Number of 32-bit values, rounded up
    static_assert(numInts <= 8, "Attribute type is too large");

    // Create an array to hold the attribute values
    uint32_t values[numInts == 0 ? 1 : numInts] = {0}; // Ensure we have at least one element

    // Read the appropriate number of attribute registers
    if constexpr (numInts > 0)
        values[0] = optixHitObjectGetAttribute_0();
    if constexpr (numInts > 1)
        values[1] = optixHitObjectGetAttribute_1();
    if constexpr (numInts > 2)
        values[2] = optixHitObjectGetAttribute_2();
    if constexpr (numInts > 3)
        values[3] = optixHitObjectGetAttribute_3();
    if constexpr (numInts > 4)
        values[4] = optixHitObjectGetAttribute_4();
    if constexpr (numInts > 5)
        values[5] = optixHitObjectGetAttribute_5();
    if constexpr (numInts > 6)
        values[6] = optixHitObjectGetAttribute_6();
    if constexpr (numInts > 7)
        values[7] = optixHitObjectGetAttribute_7();

    // Reinterpret the array as the desired type
    T result;
    memcpy(&result, values, sizeof(T));
    return result;
}

static __forceinline__ __device__ uint optixHitObjectGetSbtRecordIndex(OptixTraversableHandle* Obj)
{
    return optixHitObjectGetSbtRecordIndex();
}

static __forceinline__ __device__ uint
optixHitObjectSetSbtRecordIndex(OptixTraversableHandle* Obj, uint sbtRecordIndex)
{
    optixHitObjectSetSbtRecordIndex(sbtRecordIndex); // returns void
    return 0;
}
static __forceinline__ __device__ uint
optixHitObjectGetSbtDataPointer(OptixTraversableHandle* Obj, uint sbtRecordIndex)
{
    optixHitObjectGetSbtDataPointer(); // returns void
    return 0;
}
#endif
static const int kSlangTorchTensorMaxDim = 5;

// TensorView
struct TensorView
{
    uint8_t* data;
    uint32_t strides[kSlangTorchTensorMaxDim];
    uint32_t sizes[kSlangTorchTensorMaxDim];
    uint32_t dimensionCount;

    template<typename T>
    __device__ T* data_ptr()
    {
        return reinterpret_cast<T*>(data);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint32_t index)
    {
        uint64_t offset = strides[0] * index;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint2 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint3 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y + strides[2] * index.z;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint4 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
                          strides[3] * index.w;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T, unsigned int N>
    __device__ T* data_ptr_at(uint index[N])
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T& load(uint32_t x)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y);
    }
    template<typename T>
    __device__ T& load(uint2 index)
    {
        return *reinterpret_cast<T*>(data + strides[0] * index.x + strides[1] * index.y);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y, uint32_t z)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y + strides[2] * z);
    }
    template<typename T>
    __device__ T& load(uint3 index)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y, uint32_t z, uint32_t w)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * x + strides[1] * y + strides[2] * z + strides[3] * w);
    }
    template<typename T>
    __device__ T& load(uint4 index)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
            strides[3] * index.w);
    }
    template<typename T>
    __device__ T& load(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * i0 + strides[1] * i1 + strides[2] * i2 + strides[3] * i3 +
            strides[4] * i4);
    }

    // Generic version of load
    template<typename T, unsigned int N>
    __device__ T& load(uint index[N])
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        return *reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ void store(uint32_t x, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y) = val;
    }
    template<typename T>
    __device__ void store(uint2 index, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * index.x + strides[1] * index.y) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, uint32_t z, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y + strides[2] * z) = val;
    }
    template<typename T>
    __device__ void store(uint3 index, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, uint32_t z, uint32_t w, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * x + strides[1] * y + strides[2] * z + strides[3] * w) = val;
    }
    template<typename T>
    __device__ void store(uint4 index, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
            strides[3] * index.w) = val;
    }
    template<typename T>
    __device__ void store(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * i0 + strides[1] * i1 + strides[2] * i2 + strides[3] * i3 +
            strides[4] * i4) = val;
    }

    // Generic version
    template<typename T, unsigned int N>
    __device__ void store(uint index[N], T val)
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        *reinterpret_cast<T*>(data + offset) = val;
    }
};

// Implementations for texture fetch/load functions using tex PTX intrinsics
// These are used for read-only texture access with integer coordinates
// See #6781 for details.

// 1D is not supported via PTX. Keeping this placeholder in case it ever gets
// supported.
template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T tex1Dfetch_int(CUtexObject texObj, int x)
{
    T result;
    float stub;
    asm("tex.1d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5}];"
        : "=f"(result), "=f"(stub), "=f"(stub), "=f"(stub)
        : "l"(texObj), "r"(x));
    return result;
}

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T tex2Dfetch_int(CUtexObject texObj, int x, int y)
{
    T result;
    float stub;
    asm("tex.2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
        : "=f"(result), "=f"(stub), "=f"(stub), "=f"(stub)
        : "l"(texObj), "r"(x), "r"(y));
    return result;
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL float2 tex2Dfetch_int(CUtexObject texObj, int x, int y)
{
    float result_x, result_y;
    float stub;
    asm("tex.2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
        : "=f"(result_x), "=f"(result_y), "=f"(stub), "=f"(stub)
        : "l"(texObj), "r"(x), "r"(y));
    return make_float2(result_x, result_y);
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL float4 tex2Dfetch_int(CUtexObject texObj, int x, int y)
{
    float result_x, result_y, result_z, result_w;
    asm("tex.2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
        : "=f"(result_x), "=f"(result_y), "=f"(result_z), "=f"(result_w)
        : "l"(texObj), "r"(x), "r"(y));
    return make_float4(result_x, result_y, result_z, result_w);
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint tex2Dfetch_int(CUtexObject texObj, int x, int y)
{
    uint result;
    uint stub;
    asm("tex.2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
        : "=r"(result), "=r"(stub), "=r"(stub), "=r"(stub)
        : "l"(texObj), "r"(x), "r"(y));
    return result;
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint2 tex2Dfetch_int(CUtexObject texObj, int x, int y)
{
    uint result_x, result_y;
    uint stub;
    asm("tex.2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
        : "=r"(result_x), "=r"(result_y), "=r"(stub), "=r"(stub)
        : "l"(texObj), "r"(x), "r"(y));
    return make_uint2(result_x, result_y);
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint4 tex2Dfetch_int(CUtexObject texObj, int x, int y)
{
    uint result_x, result_y, result_z, result_w;
    asm("tex.2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
        : "=r"(result_x), "=r"(result_y), "=r"(result_z), "=r"(result_w)
        : "l"(texObj), "r"(x), "r"(y));
    return make_uint4(result_x, result_y, result_z, result_w);
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL int tex2Dfetch_int(CUtexObject texObj, int x, int y)
{
    int result;
    int stub;
    asm("tex.2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
        : "=r"(result), "=r"(stub), "=r"(stub), "=r"(stub)
        : "l"(texObj), "r"(x), "r"(y));
    return result;
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL int2 tex2Dfetch_int(CUtexObject texObj, int x, int y)
{
    int result_x, result_y;
    int stub;
    asm("tex.2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
        : "=r"(result_x), "=r"(result_y), "=r"(stub), "=r"(stub)
        : "l"(texObj), "r"(x), "r"(y));
    return make_int2(result_x, result_y);
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL int4 tex2Dfetch_int(CUtexObject texObj, int x, int y)
{
    int result_x, result_y, result_z, result_w;
    asm("tex.2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
        : "=r"(result_x), "=r"(result_y), "=r"(result_z), "=r"(result_w)
        : "l"(texObj), "r"(x), "r"(y));
    return make_int4(result_x, result_y, result_z, result_w);
}

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T tex3Dfetch_int(CUtexObject texObj, int x, int y, int z)
{
    T result;
    float stub;
    asm("tex.3d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=f"(result), "=f"(stub), "=f"(stub), "=f"(stub)
        : "l"(texObj), "r"(x), "r"(y), "r"(z), "r"(z));
    // Note: The repeated z is a stub used as the fourth operand in ptx.
    // From the docs:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-tex
    // Operand c is a scalar or singleton tuple for 1d textures; is a two-element vector for 2d
    // textures; and is a four-element vector for 3d textures.
    return result;
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL float2 tex3Dfetch_int(CUtexObject texObj, int x, int y, int z)
{
    float result_x, result_y;
    float stub;
    asm("tex.3d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=f"(result_x), "=f"(result_y), "=f"(stub), "=f"(stub)
        : "l"(texObj), "r"(x), "r"(y), "r"(z), "r"(z));
    return make_float2(result_x, result_y);
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL float4 tex3Dfetch_int(CUtexObject texObj, int x, int y, int z)
{
    float result_x, result_y, result_z, result_w;
    asm("tex.3d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=f"(result_x), "=f"(result_y), "=f"(result_z), "=f"(result_w)
        : "l"(texObj), "r"(x), "r"(y), "r"(z), "r"(z));
    return make_float4(result_x, result_y, result_z, result_w);
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint tex3Dfetch_int(CUtexObject texObj, int x, int y, int z)
{
    uint result;
    uint stub;
    asm("tex.3d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=r"(result), "=r"(stub), "=r"(stub), "=r"(stub)
        : "l"(texObj), "r"(x), "r"(y), "r"(z), "r"(z));
    return result;
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint2 tex3Dfetch_int(CUtexObject texObj, int x, int y, int z)
{
    uint result_x, result_y;
    uint stub;
    asm("tex.3d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=r"(result_x), "=r"(result_y), "=r"(stub), "=r"(stub)
        : "l"(texObj), "r"(x), "r"(y), "r"(z), "r"(z));
    return make_uint2(result_x, result_y);
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint4 tex3Dfetch_int(CUtexObject texObj, int x, int y, int z)
{
    uint result_x, result_y, result_z, result_w;
    asm("tex.3d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=r"(result_x), "=r"(result_y), "=r"(result_z), "=r"(result_w)
        : "l"(texObj), "r"(x), "r"(y), "r"(z), "r"(z));
    return make_uint4(result_x, result_y, result_z, result_w);
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL int tex3Dfetch_int(CUtexObject texObj, int x, int y, int z)
{
    int result;
    int stub;
    asm("tex.3d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=r"(result), "=r"(stub), "=r"(stub), "=r"(stub)
        : "l"(texObj), "r"(x), "r"(y), "r"(z), "r"(z));
    return result;
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL int2 tex3Dfetch_int(CUtexObject texObj, int x, int y, int z)
{
    int result_x, result_y;
    int stub;
    asm("tex.3d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=r"(result_x), "=r"(result_y), "=r"(stub), "=r"(stub)
        : "l"(texObj), "r"(x), "r"(y), "r"(z), "r"(z));
    return make_int2(result_x, result_y);
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL int4 tex3Dfetch_int(CUtexObject texObj, int x, int y, int z)
{
    int result_x, result_y, result_z, result_w;
    asm("tex.3d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=r"(result_x), "=r"(result_y), "=r"(result_z), "=r"(result_w)
        : "l"(texObj), "r"(x), "r"(y), "r"(z), "r"(z));
    return make_int4(result_x, result_y, result_z, result_w);
}

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T tex1DArrayfetch_int(CUtexObject texObj, int x, int layer)
{
    T result;
    float stub;
    asm("tex.a1d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
        : "=f"(result), "=f"(stub), "=f"(stub), "=f"(stub)
        : "l"(texObj), "r"(x), "r"(layer));
    return result;
}

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T
tex2DArrayfetch_int(CUtexObject texObj, int x, int y, int layer)
{
    T result;
    float stub;
    asm("tex.a2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=f"(result), "=f"(stub), "=f"(stub), "=f"(stub)
        : "l"(texObj), "r"(x), "r"(y), "r"(layer), "r"(layer));
    return result;
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL float2
tex2DArrayfetch_int(CUtexObject texObj, int x, int y, int layer)
{
    float result_x, result_y;
    float stub;
    asm("tex.a2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=f"(result_x), "=f"(result_y), "=f"(stub), "=f"(stub)
        : "l"(texObj), "r"(x), "r"(y), "r"(layer), "r"(layer));
    return make_float2(result_x, result_y);
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL float4
tex2DArrayfetch_int(CUtexObject texObj, int x, int y, int layer)
{
    float result_x, result_y, result_z, result_w;
    asm("tex.a2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=f"(result_x), "=f"(result_y), "=f"(result_z), "=f"(result_w)
        : "l"(texObj), "r"(x), "r"(y), "r"(layer), "r"(layer));
    return make_float4(result_x, result_y, result_z, result_w);
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint
tex2DArrayfetch_int(CUtexObject texObj, int x, int y, int layer)
{
    uint result;
    uint stub;
    asm("tex.a2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=r"(result), "=r"(stub), "=r"(stub), "=r"(stub)
        : "l"(texObj), "r"(x), "r"(y), "r"(layer), "r"(layer));
    return result;
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint2
tex2DArrayfetch_int(CUtexObject texObj, int x, int y, int layer)
{
    uint result_x, result_y;
    uint stub;
    asm("tex.a2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=r"(result_x), "=r"(result_y), "=r"(stub), "=r"(stub)
        : "l"(texObj), "r"(x), "r"(y), "r"(layer), "r"(layer));
    return make_uint2(result_x, result_y);
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint4
tex2DArrayfetch_int(CUtexObject texObj, int x, int y, int layer)
{
    uint result_x, result_y, result_z, result_w;
    asm("tex.a2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=r"(result_x), "=r"(result_y), "=r"(result_z), "=r"(result_w)
        : "l"(texObj), "r"(x), "r"(y), "r"(layer), "r"(layer));
    return make_uint4(result_x, result_y, result_z, result_w);
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL int tex2DArrayfetch_int(
    CUtexObject texObj,
    int x,
    int y,
    int layer)
{
    int result;
    int stub;
    asm("tex.a2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=r"(result), "=r"(stub), "=r"(stub), "=r"(stub)
        : "l"(texObj), "r"(x), "r"(y), "r"(layer), "r"(layer));
    return result;
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL int2
tex2DArrayfetch_int(CUtexObject texObj, int x, int y, int layer)
{
    int result_x, result_y;
    int stub;
    asm("tex.a2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=r"(result_x), "=r"(result_y), "=r"(stub), "=r"(stub)
        : "l"(texObj), "r"(x), "r"(y), "r"(layer), "r"(layer));
    return make_int2(result_x, result_y);
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL int4
tex2DArrayfetch_int(CUtexObject texObj, int x, int y, int layer)
{
    int result_x, result_y, result_z, result_w;
    asm("tex.a2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=r"(result_x), "=r"(result_y), "=r"(result_z), "=r"(result_w)
        : "l"(texObj), "r"(x), "r"(y), "r"(layer), "r"(layer));
    return make_int4(result_x, result_y, result_z, result_w);
}


#line 234 "/data/fwindisch/hierarchical-LOD-gaussians/submodules/lod-slang-gaussian-rasterization/lod_slang_gaussian_rasterization/internal/slang/utils.slang"
struct Splat_2D_AlphaBlend_0
{
    float3  xyz_vs_0;
    float3  rgb_0;
    float opacity_0;
    Matrix<float, 2, 2>  inv_cov_vs_0;
    float distance_mu_0;
    float distance_sigma_0;
};


#line 1199 "core.meta.slang"
__device__ Splat_2D_AlphaBlend_0 Splat_2D_AlphaBlend_x24_syn_dzero_0()
{

#line 1199
    Splat_2D_AlphaBlend_0 result_0;

#line 2059
    float3  _S1 = make_float3 (0.0f);

#line 2059
    (&result_0)->xyz_vs_0 = _S1;

#line 2059
    (&result_0)->rgb_0 = _S1;

#line 2059
    (&result_0)->opacity_0 = 0.0f;

#line 2059
    (&result_0)->inv_cov_vs_0 = makeMatrix<float, 2, 2> (0.0f);

#line 2059
    (&result_0)->distance_mu_0 = 0.0f;

#line 2059
    (&result_0)->distance_sigma_0 = 0.0f;

#line 2059
    return result_0;
}


#line 843 "diff.meta.slang"
struct AtomicAdd_0
{
    TensorView diff_0;
};


#line 958
struct DiffTensorView_0
{
    TensorView primal_0;
    AtomicAdd_0 diff_1;
};


#line 963
__device__ uint DiffTensorView_size_0(DiffTensorView_0 this_0, uint i_0)
{
    uint _S2 = ((this_0.primal_0).sizes[(i_0)]);

#line 965
    return _S2;
}


#line 999
__device__ float DiffTensorView_load_0(DiffTensorView_0 this_1, uint3  i_1)
{

#line 999
    float _S3 = ((this_1.primal_0).load<float>((i_1)));

#line 999
    return _S3;
}


#line 999
__device__ float DiffTensorView_load_1(DiffTensorView_0 this_2, uint2  i_2)
{

#line 999
    float _S4 = ((this_2.primal_0).load<float>((i_2)));

#line 999
    return _S4;
}


#line 23 "/data/fwindisch/hierarchical-LOD-gaussians/submodules/lod-slang-gaussian-rasterization/lod_slang_gaussian_rasterization/internal/slang/alphablend_shader.slang"
__device__ __shared__ FixedArray<uint, 64>  collected_idx_0;


#line 22
__device__ __shared__ FixedArray<Splat_2D_AlphaBlend_0, 64>  collected_splats_0;


#line 26 "/data/fwindisch/hierarchical-LOD-gaussians/submodules/lod-slang-gaussian-rasterization/lod_slang_gaussian_rasterization/internal/slang/utils.slang"
__device__ float3  read_t3_float3_0(uint idx_0, DiffTensorView_0 t3_0)
{
    return make_float3 (DiffTensorView_load_1(t3_0, make_uint2 (idx_0, 0U)), DiffTensorView_load_1(t3_0, make_uint2 (idx_0, 1U)), DiffTensorView_load_1(t3_0, make_uint2 (idx_0, 2U)));
}


#line 20
__device__ float read_t1_float_0(uint idx_1, DiffTensorView_0 t1_0)
{
    return DiffTensorView_load_1(t1_0, make_uint2 (idx_1, 0U));
}


#line 52
__device__ Matrix<float, 2, 2>  read_t2x2_float2x2_0(uint idx_2, DiffTensorView_0 t2x2_0)
{
    return makeMatrix<float, 2, 2> (DiffTensorView_load_0(t2x2_0, make_uint3 (idx_2, 0U, 0U)), DiffTensorView_load_0(t2x2_0, make_uint3 (idx_2, 1U, 0U)), DiffTensorView_load_0(t2x2_0, make_uint3 (idx_2, 0U, 1U)), DiffTensorView_load_0(t2x2_0, make_uint3 (idx_2, 1U, 1U)));
}


#line 234
__device__ Splat_2D_AlphaBlend_0 Splat_2D_AlphaBlend_x24init_0(float3  xyz_vs_1, float3  rgb_1, float opacity_1, Matrix<float, 2, 2>  inv_cov_vs_1, float distance_mu_1, float distance_sigma_1)
{

#line 234
    Splat_2D_AlphaBlend_0 _S5;

    (&_S5)->xyz_vs_0 = xyz_vs_1;
    (&_S5)->rgb_0 = rgb_1;
    (&_S5)->opacity_0 = opacity_1;
    (&_S5)->inv_cov_vs_0 = inv_cov_vs_1;
    (&_S5)->distance_mu_0 = distance_mu_1;
    (&_S5)->distance_sigma_0 = distance_sigma_1;

#line 234
    return _S5;
}


#line 245
__device__ Splat_2D_AlphaBlend_0 load_splat_alphablend_0(int g_idx_0, DiffTensorView_0 xyz_vs_2, DiffTensorView_0 inv_cov_vs_2, DiffTensorView_0 opacity_2, DiffTensorView_0 rgb_2, DiffTensorView_0 distance_mu_2, DiffTensorView_0 distance_sigma_2)
{

#line 253
    uint _S6 = uint(g_idx_0);

#line 261
    return Splat_2D_AlphaBlend_x24init_0(read_t3_float3_0(_S6, xyz_vs_2), read_t3_float3_0(_S6, rgb_2), read_t1_float_0(_S6, opacity_2), read_t2x2_float2x2_0(_S6, inv_cov_vs_2), read_t1_float_0(_S6, distance_mu_2), read_t1_float_0(_S6, distance_sigma_2));
}


#line 61
__device__ float ndc2pix_0(float v_0, int S_0)
{
    return ((v_0 + 1.0f) * float(S_0) - 1.0f) * 0.5f;
}


#line 63
struct DiffPair_float_0
{
    float primal_1;
    float differential_0;
};


#line 1 "token paste"
__device__ void _d_exp_0(DiffPair_float_0 * dpx_0, float dOut_0)
{

#line 1907 "diff.meta.slang"
    float _S7 = (F32_exp(((*dpx_0).primal_1))) * dOut_0;

#line 1907
    dpx_0->primal_1 = (*dpx_0).primal_1;

#line 1907
    dpx_0->differential_0 = _S7;



    return;
}


#line 1 "token paste"
__device__ DiffPair_float_0 _d_exp_1(DiffPair_float_0 dpx_1)
{

#line 1880 "diff.meta.slang"
    float _S8 = (F32_exp((dpx_1.primal_1)));

#line 1880
    DiffPair_float_0 _S9 = { _S8, _S8 * dpx_1.differential_0 };

#line 1880
    return _S9;
}


#line 2154
__device__ void _d_min_0(DiffPair_float_0 * dpx_2, DiffPair_float_0 * dpy_0, float dOut_1)
{
    DiffPair_float_0 _S10 = *dpx_2;

#line 2156
    float _S11;

#line 2156
    if(((*dpx_2).primal_1) < ((*dpy_0).primal_1))
    {

#line 2156
        _S11 = dOut_1;

#line 2156
    }
    else
    {

#line 2156
        if(((*dpx_2).primal_1) > ((*dpy_0).primal_1))
        {

#line 2156
            _S11 = 0.0f;

#line 2156
        }
        else
        {

#line 2156
            _S11 = 0.5f * dOut_1;

#line 2156
        }

#line 2156
    }

#line 2156
    dpx_2->primal_1 = _S10.primal_1;

#line 2156
    dpx_2->differential_0 = _S11;
    DiffPair_float_0 _S12 = *dpy_0;

#line 2157
    if(((*dpy_0).primal_1) < (_S10.primal_1))
    {

#line 2157
        _S11 = dOut_1;

#line 2157
    }
    else
    {

#line 2157
        if(((*dpy_0).primal_1) > ((*dpx_2).primal_1))
        {

#line 2157
            _S11 = 0.0f;

#line 2157
        }
        else
        {

#line 2157
            _S11 = 0.5f * dOut_1;

#line 2157
        }

#line 2157
    }

#line 2157
    dpy_0->primal_1 = _S12.primal_1;

#line 2157
    dpy_0->differential_0 = _S11;
    return;
}


#line 2142
__device__ DiffPair_float_0 _d_min_1(DiffPair_float_0 dpx_3, DiffPair_float_0 dpy_1)
{

    float _S13 = (F32_min((dpx_3.primal_1), (dpy_1.primal_1)));

#line 2145
    float _S14;
    if((dpx_3.primal_1) < (dpy_1.primal_1))
    {

#line 2146
        _S14 = dpx_3.differential_0;

#line 2146
    }
    else
    {

#line 2146
        if((dpx_3.primal_1) > (dpy_1.primal_1))
        {

#line 2146
            _S14 = dpy_1.differential_0;

#line 2146
        }
        else
        {

#line 2146
            _S14 = 0.5f * (dpx_3.differential_0 + dpy_1.differential_0);

#line 2146
        }

#line 2146
    }

#line 2146
    DiffPair_float_0 _S15 = { _S13, _S14 };

#line 2144
    return _S15;
}


#line 289 "/data/fwindisch/hierarchical-LOD-gaussians/submodules/lod-slang-gaussian-rasterization/lod_slang_gaussian_rasterization/internal/slang/utils.slang"
__device__ float4  evaluate_splat_0(Splat_2D_AlphaBlend_0 g_0, float2  pix_coord_0, float distance_0, uint H_0, uint W_0)
{

#line 301
    float _S16 = pix_coord_0.x - ndc2pix_0(g_0.xyz_vs_0.x, int(W_0));
    float _S17 = pix_coord_0.y - ndc2pix_0(g_0.xyz_vs_0.y, int(H_0));



    float _S18 = distance_0 - g_0.distance_mu_0;

#line 306
    float _S19 = g_0.distance_sigma_0;

#line 306
    float alpha_0 = (F32_min((0.99000000953674316f), (g_0.opacity_0 * (F32_exp((-0.5f * (g_0.inv_cov_vs_0.rows[int(0)].x * _S16 * _S16 + g_0.inv_cov_vs_0.rows[int(1)].y * _S17 * _S17 + (g_0.inv_cov_vs_0.rows[int(0)].y + g_0.inv_cov_vs_0.rows[int(1)].x) * _S16 * _S17))))))) * (F32_exp((-0.5f * (_S18 * _S18) / (_S19 * _S19 + 1.00000001168609742e-07f))));


    return make_float4 ((g_0.rgb_0 * make_float3 (alpha_0)).x, (g_0.rgb_0 * make_float3 (alpha_0)).y, (g_0.rgb_0 * make_float3 (alpha_0)).z, alpha_0);
}


#line 34 "/data/fwindisch/hierarchical-LOD-gaussians/submodules/lod-slang-gaussian-rasterization/lod_slang_gaussian_rasterization/internal/slang/alphablend_shader.slang"
__device__ float4  undo_pixel_state_0(float4  pixel_state_t_n_0, float4  gauss_rgba_t_n_0)
{
    float transmittance_t_nm1_0 = pixel_state_t_n_0.w / (1.0f - gauss_rgba_t_n_0.w);

    return make_float4 ((float3 {pixel_state_t_n_0.x, pixel_state_t_n_0.y, pixel_state_t_n_0.z} - float3 {gauss_rgba_t_n_0.x, gauss_rgba_t_n_0.y, gauss_rgba_t_n_0.z} * make_float3 (transmittance_t_nm1_0)).x, (float3 {pixel_state_t_n_0.x, pixel_state_t_n_0.y, pixel_state_t_n_0.z} - float3 {gauss_rgba_t_n_0.x, gauss_rgba_t_n_0.y, gauss_rgba_t_n_0.z} * make_float3 (transmittance_t_nm1_0)).y, (float3 {pixel_state_t_n_0.x, pixel_state_t_n_0.y, pixel_state_t_n_0.z} - float3 {gauss_rgba_t_n_0.x, gauss_rgba_t_n_0.y, gauss_rgba_t_n_0.z} * make_float3 (transmittance_t_nm1_0)).z, transmittance_t_nm1_0);
}


#line 27
__device__ float4  update_pixel_state_0(float4  pixel_state_t_nm1_0, float4  gauss_rgba_t_n_1)
{
    float _S20 = pixel_state_t_nm1_0.w;

    return make_float4 ((float3 {pixel_state_t_nm1_0.x, pixel_state_t_nm1_0.y, pixel_state_t_nm1_0.z} + float3 {gauss_rgba_t_n_1.x, gauss_rgba_t_n_1.y, gauss_rgba_t_n_1.z} * make_float3 (_S20)).x, (float3 {pixel_state_t_nm1_0.x, pixel_state_t_nm1_0.y, pixel_state_t_nm1_0.z} + float3 {gauss_rgba_t_n_1.x, gauss_rgba_t_n_1.y, gauss_rgba_t_n_1.z} * make_float3 (_S20)).y, (float3 {pixel_state_t_nm1_0.x, pixel_state_t_nm1_0.y, pixel_state_t_nm1_0.z} + float3 {gauss_rgba_t_n_1.x, gauss_rgba_t_n_1.y, gauss_rgba_t_n_1.z} * make_float3 (_S20)).z, _S20 * (1.0f - gauss_rgba_t_n_1.w));
}


#line 268
struct DiffPair_vectorx3Cfloatx2C2x3E_0
{
    float2  primal_1;
    float2  differential_0;
};


#line 79
struct DiffPair_Splat_2D_AlphaBlend_0
{
    Splat_2D_AlphaBlend_0 primal_1;
    Splat_2D_AlphaBlend_0 differential_0;
};


#line 197
struct DiffPair_vectorx3Cfloatx2C4x3E_0
{
    float4  primal_1;
    float4  differential_0;
};


#line 27
__device__ void s_bwd_prop_update_pixel_state_0(DiffPair_vectorx3Cfloatx2C4x3E_0 * dppixel_state_t_nm1_0, DiffPair_vectorx3Cfloatx2C4x3E_0 * dpgauss_rgba_t_n_0, float4  _s_dOut_0)
{
    float _S21 = (*dppixel_state_t_nm1_0).primal_1.w;

    float3  s_diff_color_t_n_T_0 = float3 {_s_dOut_0.x, _s_dOut_0.y, _s_dOut_0.z};

#line 29
    float3  _S22 = float3 {(*dpgauss_rgba_t_n_0).primal_1.x, (*dpgauss_rgba_t_n_0).primal_1.y, (*dpgauss_rgba_t_n_0).primal_1.z} * s_diff_color_t_n_T_0;

#line 29
    float3  _S23 = make_float3 (_S21) * s_diff_color_t_n_T_0;

#line 29
    float _S24 = (1.0f - (*dpgauss_rgba_t_n_0).primal_1.w) * _s_dOut_0.w + _S22.x + _S22.y + _S22.z;

#line 29
    float4  _S25 = make_float4 (_S23.x, _S23.y, _S23.z, - (_S21 * _s_dOut_0.w));

#line 29
    dpgauss_rgba_t_n_0->primal_1 = (*dpgauss_rgba_t_n_0).primal_1;

#line 29
    dpgauss_rgba_t_n_0->differential_0 = _S25;

#line 29
    float4  _S26 = make_float4 (s_diff_color_t_n_T_0.x, s_diff_color_t_n_T_0.y, s_diff_color_t_n_T_0.z, _S24);

#line 29
    dppixel_state_t_nm1_0->primal_1 = (*dppixel_state_t_nm1_0).primal_1;

#line 29
    dppixel_state_t_nm1_0->differential_0 = _S26;

#line 27
    return;
}


#line 27
__device__ void s_bwd_update_pixel_state_0(DiffPair_vectorx3Cfloatx2C4x3E_0 * _S27, DiffPair_vectorx3Cfloatx2C4x3E_0 * _S28, float4  _S29)
{

#line 27
    s_bwd_prop_update_pixel_state_0(_S27, _S28, _S29);

#line 27
    return;
}


#line 211
__device__ float s_primal_ctx_ndc2pix_0(float dpv_0, int S_1)
{

#line 61 "/data/fwindisch/hierarchical-LOD-gaussians/submodules/lod-slang-gaussian-rasterization/lod_slang_gaussian_rasterization/internal/slang/utils.slang"
    return ((dpv_0 + 1.0f) * float(S_1) - 1.0f) * 0.5f;
}


#line 61
__device__ float s_primal_ctx_exp_0(float _S30)
{

#line 61
    return (F32_exp((_S30)));
}


#line 61
__device__ float s_primal_ctx_min_0(float _S31, float _S32)
{

#line 61
    return (F32_min((_S31), (_S32)));
}


#line 61
__device__ void s_bwd_prop_exp_0(DiffPair_float_0 * _S33, float _S34)
{

#line 61
    _d_exp_0(_S33, _S34);

#line 61
    return;
}


#line 61
__device__ void s_bwd_prop_min_0(DiffPair_float_0 * _S35, DiffPair_float_0 * _S36, float _S37)
{

#line 61
    _d_min_0(_S35, _S36, _S37);

#line 61
    return;
}


#line 61
__device__ void s_bwd_prop_ndc2pix_0(DiffPair_float_0 * dpv_1, int S_2, float _s_dOut_1)
{
    float _S38 = float(S_2) * (0.5f * _s_dOut_1);

#line 63
    dpv_1->primal_1 = (*dpv_1).primal_1;

#line 63
    dpv_1->differential_0 = _S38;

#line 61
    return;
}


#line 289
__device__ void s_bwd_prop_evaluate_splat_0(DiffPair_Splat_2D_AlphaBlend_0 * dpg_0, DiffPair_vectorx3Cfloatx2C2x3E_0 * dppix_coord_0, DiffPair_float_0 * dpdistance_0, uint H_1, uint W_1, float4  _s_dOut_2)
{

#line 301
    float _S39 = (*dpg_0).primal_1.xyz_vs_0.x;

#line 301
    int _S40 = int(W_1);

#line 301
    float _S41 = (*dppix_coord_0).primal_1.x - s_primal_ctx_ndc2pix_0(_S39, _S40);
    float _S42 = (*dpg_0).primal_1.xyz_vs_0.y;

#line 302
    int _S43 = int(H_1);

#line 302
    float _S44 = (*dppix_coord_0).primal_1.y - s_primal_ctx_ndc2pix_0(_S42, _S43);
    float _S45 = (*dpg_0).primal_1.inv_cov_vs_0.rows[int(0)].x * _S41;
    float _S46 = (*dpg_0).primal_1.inv_cov_vs_0.rows[int(1)].y * _S44;

#line 304
    float _S47 = (*dpg_0).primal_1.inv_cov_vs_0.rows[int(0)].y + (*dpg_0).primal_1.inv_cov_vs_0.rows[int(1)].x;

#line 304
    float _S48 = _S47 * _S41;

#line 303
    float power_0 = -0.5f * (_S45 * _S41 + _S46 * _S44 + _S48 * _S44);

#line 303
    float _S49 = s_primal_ctx_exp_0(power_0);

    float _S50 = (*dpg_0).primal_1.opacity_0 * _S49;

#line 305
    float _S51 = s_primal_ctx_min_0(0.99000000953674316f, _S50);
    float _S52 = (*dpdistance_0).primal_1 - (*dpg_0).primal_1.distance_mu_0;

#line 306
    float _S53 = -0.5f * (_S52 * _S52);

#line 306
    float _S54 = (*dpg_0).primal_1.distance_sigma_0;

#line 306
    float _S55 = _S54 * _S54 + 1.00000001168609742e-07f;

#line 306
    float _S56 = _S53 / _S55;

#line 306
    float _S57 = _S55 * _S55;

#line 306
    float _S58 = s_primal_ctx_exp_0(_S56);


    float3  s_diff_premult_rgb_T_0 = float3 {_s_dOut_2.x, _s_dOut_2.y, _s_dOut_2.z};

#line 307
    float3  _S59 = (*dpg_0).primal_1.rgb_0 * s_diff_premult_rgb_T_0;

#line 307
    float3  _S60 = make_float3 (_S51 * _S58) * s_diff_premult_rgb_T_0;

#line 306
    float _S61 = _s_dOut_2.w + _S59.x + _S59.y + _S59.z;

#line 306
    float _S62 = _S51 * _S61;

#line 306
    float _S63 = _S58 * _S61;

#line 306
    DiffPair_float_0 _S64;

#line 306
    (&_S64)->primal_1 = _S56;

#line 306
    (&_S64)->differential_0 = 0.0f;

#line 306
    s_bwd_prop_exp_0(&_S64, _S62);

#line 306
    float _S65 = _S64.differential_0 / _S57;

#line 306
    float _S66 = (*dpg_0).primal_1.distance_sigma_0 * (_S53 * - _S65);

#line 1201 "core.meta.slang"
    float _S67 = _S66 + _S66;

#line 306 "/data/fwindisch/hierarchical-LOD-gaussians/submodules/lod-slang-gaussian-rasterization/lod_slang_gaussian_rasterization/internal/slang/utils.slang"
    float _S68 = _S52 * (-0.5f * (_S55 * _S65));

#line 306
    float _S69 = _S68 + _S68;

#line 306
    float _S70 = - _S69;

#line 305
    DiffPair_float_0 _S71;

#line 305
    (&_S71)->primal_1 = 0.99000000953674316f;

#line 305
    (&_S71)->differential_0 = 0.0f;

#line 305
    DiffPair_float_0 _S72;

#line 305
    (&_S72)->primal_1 = _S50;

#line 305
    (&_S72)->differential_0 = 0.0f;

#line 305
    s_bwd_prop_min_0(&_S71, &_S72, _S63);

#line 305
    float _S73 = (*dpg_0).primal_1.opacity_0 * _S72.differential_0;

#line 305
    float _S74 = _S49 * _S72.differential_0;

#line 305
    DiffPair_float_0 _S75;

#line 305
    (&_S75)->primal_1 = power_0;

#line 305
    (&_S75)->differential_0 = 0.0f;

#line 305
    s_bwd_prop_exp_0(&_S75, _S73);

#line 303
    float _S76 = -0.5f * _S75.differential_0;
    float _S77 = _S48 * _S76;

#line 304
    float _S78 = _S44 * _S76;

#line 304
    float _S79 = _S47 * _S78;

#line 304
    float _S80 = _S41 * _S78;

#line 304
    float _S81 = _S46 * _S76;

#line 304
    float _S82 = (*dpg_0).primal_1.inv_cov_vs_0.rows[int(1)].y * _S78;

#line 304
    float _S83 = _S44 * _S78;

#line 2059 "core.meta.slang"
    float2  _S84 = make_float2 (0.0f);

#line 2059
    float2  _S85 = _S84;

#line 2059
    *&((&_S85)->x) = _S80;

#line 2059
    *&((&_S85)->y) = _S83;

#line 303 "/data/fwindisch/hierarchical-LOD-gaussians/submodules/lod-slang-gaussian-rasterization/lod_slang_gaussian_rasterization/internal/slang/utils.slang"
    float _S86 = _S45 * _S76;

#line 303
    float _S87 = _S41 * _S76;

#line 303
    float _S88 = (*dpg_0).primal_1.inv_cov_vs_0.rows[int(0)].x * _S87;

#line 303
    float _S89 = _S41 * _S87;

#line 303
    float2  _S90 = _S84;

#line 303
    *&((&_S90)->y) = _S80;

#line 303
    *&((&_S90)->x) = _S89;

#line 302
    float _S91 = _S77 + _S81 + _S82;

#line 302
    float _S92 = - _S91;

#line 302
    DiffPair_float_0 _S93;

#line 302
    (&_S93)->primal_1 = _S42;

#line 302
    (&_S93)->differential_0 = 0.0f;

#line 302
    s_bwd_prop_ndc2pix_0(&_S93, _S43, _S92);

#line 301
    float _S94 = _S79 + _S86 + _S88;

#line 301
    float _S95 = - _S94;

#line 301
    DiffPair_float_0 _S96;

#line 301
    (&_S96)->primal_1 = _S39;

#line 301
    (&_S96)->differential_0 = 0.0f;

#line 301
    s_bwd_prop_ndc2pix_0(&_S96, _S40, _S95);

#line 301
    Matrix<float, 2, 2>  _S97 = makeMatrix<float, 2, 2> (0.0f);

#line 301
    _S97[int(1)] = _S85;

#line 301
    _S97[int(0)] = _S90;

#line 301
    float3  _S98 = make_float3 (_S96.differential_0, _S93.differential_0, 0.0f);

#line 301
    dpdistance_0->primal_1 = (*dpdistance_0).primal_1;

#line 301
    dpdistance_0->differential_0 = _S69;

#line 301
    float2  _S99 = make_float2 (_S94, _S91);

#line 301
    dppix_coord_0->primal_1 = (*dppix_coord_0).primal_1;

#line 301
    dppix_coord_0->differential_0 = _S99;

#line 301
    Splat_2D_AlphaBlend_0 _S100 = Splat_2D_AlphaBlend_x24_syn_dzero_0();

#line 301
    (&_S100)->distance_sigma_0 = _S67;

#line 301
    (&_S100)->distance_mu_0 = _S70;

#line 301
    (&_S100)->inv_cov_vs_0 = _S97;

#line 301
    (&_S100)->opacity_0 = _S74;

#line 301
    (&_S100)->rgb_0 = _S60;

#line 301
    (&_S100)->xyz_vs_0 = _S98;

#line 301
    dpg_0->primal_1 = (*dpg_0).primal_1;

#line 301
    dpg_0->differential_0 = _S100;

#line 289
    return;
}


#line 289
__device__ void s_bwd_evaluate_splat_0(DiffPair_Splat_2D_AlphaBlend_0 * _S101, DiffPair_vectorx3Cfloatx2C2x3E_0 * _S102, DiffPair_float_0 * _S103, uint _S104, uint _S105, float4  _S106)
{


    s_bwd_prop_evaluate_splat_0(_S101, _S102, _S103, _S104, _S105, _S106);

#line 293
    return;
}


#line 245
struct s_bwd_prop_load_splat_alphablend_Intermediates_0
{
    float3  _S107;
    float3  _S108;
    float _S109;
    Matrix<float, 2, 2>  _S110;
    float _S111;
    float _S112;
};


#line 245
__device__ float3  s_primal_ctx_read_t3_float3_0(uint idx_3, DiffTensorView_0 t3_1)
{

#line 26
    return make_float3 (DiffTensorView_load_1(t3_1, make_uint2 (idx_3, 0U)), DiffTensorView_load_1(t3_1, make_uint2 (idx_3, 1U)), DiffTensorView_load_1(t3_1, make_uint2 (idx_3, 2U)));
}


#line 26
__device__ float s_primal_ctx_read_t1_float_0(uint idx_4, DiffTensorView_0 t1_1)
{

#line 20
    return DiffTensorView_load_1(t1_1, make_uint2 (idx_4, 0U));
}


#line 20
__device__ Matrix<float, 2, 2>  s_primal_ctx_read_t2x2_float2x2_0(uint idx_5, DiffTensorView_0 t2x2_1)
{

#line 52
    return makeMatrix<float, 2, 2> (DiffTensorView_load_0(t2x2_1, make_uint3 (idx_5, 0U, 0U)), DiffTensorView_load_0(t2x2_1, make_uint3 (idx_5, 1U, 0U)), DiffTensorView_load_0(t2x2_1, make_uint3 (idx_5, 0U, 1U)), DiffTensorView_load_0(t2x2_1, make_uint3 (idx_5, 1U, 1U)));
}


#line 52
__device__ Splat_2D_AlphaBlend_0 s_primal_ctx_Splat_2D_AlphaBlend_x24init_0(float3  dpxyz_vs_0, float3  dprgb_0, float dpopacity_0, Matrix<float, 2, 2>  dpinv_cov_vs_0, float dpdistance_mu_0, float dpdistance_sigma_0)
{

#line 241
    Splat_2D_AlphaBlend_0 _S113 = { dpxyz_vs_0, dprgb_0, dpopacity_0, dpinv_cov_vs_0, dpdistance_mu_0, dpdistance_sigma_0 };

#line 241
    return _S113;
}


#line 241
__device__ Splat_2D_AlphaBlend_0 s_primal_ctx_load_splat_alphablend_0(int g_idx_1, DiffTensorView_0 xyz_vs_3, DiffTensorView_0 inv_cov_vs_3, DiffTensorView_0 opacity_3, DiffTensorView_0 rgb_3, DiffTensorView_0 distance_mu_3, DiffTensorView_0 distance_sigma_3, s_bwd_prop_load_splat_alphablend_Intermediates_0 * _s_diff_ctx_0)
{

#line 251
    float3  _S114 = make_float3 (0.0f);

#line 251
    Matrix<float, 2, 2>  _S115 = makeMatrix<float, 2, 2> (0.0f);

#line 251
    _s_diff_ctx_0->_S107 = _S114;

#line 251
    _s_diff_ctx_0->_S108 = _S114;

#line 251
    _s_diff_ctx_0->_S109 = 0.0f;

#line 251
    _s_diff_ctx_0->_S110 = _S115;

#line 251
    _s_diff_ctx_0->_S111 = 0.0f;

#line 251
    _s_diff_ctx_0->_S112 = 0.0f;

    _s_diff_ctx_0->_S107 = _S114;
    _s_diff_ctx_0->_S108 = _S114;
    _s_diff_ctx_0->_S109 = 0.0f;
    _s_diff_ctx_0->_S110 = _S115;

    _s_diff_ctx_0->_S111 = 0.0f;
    _s_diff_ctx_0->_S112 = 0.0f;

#line 253
    uint _S116 = uint(g_idx_1);

#line 253
    float3  _S117 = s_primal_ctx_read_t3_float3_0(_S116, xyz_vs_3);

#line 253
    _s_diff_ctx_0->_S107 = _S117;

#line 253
    float3  _S118 = s_primal_ctx_read_t3_float3_0(_S116, rgb_3);
    _s_diff_ctx_0->_S108 = _S118;

#line 254
    float _S119 = s_primal_ctx_read_t1_float_0(_S116, opacity_3);
    _s_diff_ctx_0->_S109 = _S119;

#line 255
    Matrix<float, 2, 2>  _S120 = s_primal_ctx_read_t2x2_float2x2_0(_S116, inv_cov_vs_3);
    _s_diff_ctx_0->_S110 = _S120;

#line 256
    float _S121 = s_primal_ctx_read_t1_float_0(_S116, distance_mu_3);

    _s_diff_ctx_0->_S111 = _S121;

#line 258
    float _S122 = s_primal_ctx_read_t1_float_0(_S116, distance_sigma_3);
    _s_diff_ctx_0->_S112 = _S122;

#line 259
    return s_primal_ctx_Splat_2D_AlphaBlend_x24init_0(_S117, _S118, _S119, _S120, _S121, _S122);
}


#line 259
struct DiffPair_vectorx3Cfloatx2C3x3E_0
{
    float3  primal_1;
    float3  differential_0;
};


#line 261
struct DiffPair_matrixx3Cfloatx2C2x2C2x3E_0
{
    Matrix<float, 2, 2>  primal_1;
    Matrix<float, 2, 2>  differential_0;
};


#line 234
__device__ void s_bwd_prop_Splat_2D_AlphaBlend_x24init_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpxyz_vs_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dprgb_1, DiffPair_float_0 * dpopacity_1, DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 * dpinv_cov_vs_1, DiffPair_float_0 * dpdistance_mu_1, DiffPair_float_0 * dpdistance_sigma_1, Splat_2D_AlphaBlend_0 _s_dOut_3)
{

#line 234
    dpdistance_sigma_1->primal_1 = (*dpdistance_sigma_1).primal_1;

#line 234
    dpdistance_sigma_1->differential_0 = _s_dOut_3.distance_sigma_0;

#line 234
    dpdistance_mu_1->primal_1 = (*dpdistance_mu_1).primal_1;

#line 234
    dpdistance_mu_1->differential_0 = _s_dOut_3.distance_mu_0;

#line 234
    dpinv_cov_vs_1->primal_1 = (*dpinv_cov_vs_1).primal_1;

#line 234
    dpinv_cov_vs_1->differential_0 = _s_dOut_3.inv_cov_vs_0;

#line 234
    dpopacity_1->primal_1 = (*dpopacity_1).primal_1;

#line 234
    dpopacity_1->differential_0 = _s_dOut_3.opacity_0;

#line 234
    dprgb_1->primal_1 = (*dprgb_1).primal_1;

#line 234
    dprgb_1->differential_0 = _s_dOut_3.rgb_0;

#line 234
    dpxyz_vs_1->primal_1 = (*dpxyz_vs_1).primal_1;

#line 234
    dpxyz_vs_1->differential_0 = _s_dOut_3.xyz_vs_0;

#line 234
    return;
}


#line 869 "diff.meta.slang"
__device__ void AtomicAdd_load_backward_0(AtomicAdd_0 this_3, uint2  i_3, float dOut_2)
{
    float oldVal_0;
    *((&oldVal_0)) = atomicAdd((this_3.diff_0).data_ptr_at<float>((i_3)), (dOut_2));
    return;
}


#line 20 "/data/fwindisch/hierarchical-LOD-gaussians/submodules/lod-slang-gaussian-rasterization/lod_slang_gaussian_rasterization/internal/slang/utils.slang"
__device__ void s_bwd_prop_read_t1_float_0(uint idx_6, DiffTensorView_0 t1_2, float _s_dOut_4)
{
    AtomicAdd_load_backward_0(t1_2.diff_1, make_uint2 (idx_6, 0U), _s_dOut_4);

#line 20
    return;
}


#line 869 "diff.meta.slang"
__device__ void AtomicAdd_load_backward_1(AtomicAdd_0 this_4, uint3  i_4, float dOut_3)
{
    float oldVal_1;
    *((&oldVal_1)) = atomicAdd((this_4.diff_0).data_ptr_at<float>((i_4)), (dOut_3));
    return;
}


#line 52 "/data/fwindisch/hierarchical-LOD-gaussians/submodules/lod-slang-gaussian-rasterization/lod_slang_gaussian_rasterization/internal/slang/utils.slang"
__device__ void s_bwd_prop_read_t2x2_float2x2_0(uint idx_7, DiffTensorView_0 t2x2_2, Matrix<float, 2, 2>  _s_dOut_5)
{
    uint3  _S123 = make_uint3 (idx_7, 0U, 0U);
    uint3  _S124 = make_uint3 (idx_7, 1U, 0U);
    uint3  _S125 = make_uint3 (idx_7, 0U, 1U);

#line 54
    AtomicAdd_load_backward_1(t2x2_2.diff_1, make_uint3 (idx_7, 1U, 1U), _s_dOut_5.rows[int(1)].y);

#line 54
    AtomicAdd_load_backward_1(t2x2_2.diff_1, _S125, _s_dOut_5.rows[int(1)].x);

#line 54
    AtomicAdd_load_backward_1(t2x2_2.diff_1, _S124, _s_dOut_5.rows[int(0)].y);

#line 54
    AtomicAdd_load_backward_1(t2x2_2.diff_1, _S123, _s_dOut_5.rows[int(0)].x);

#line 52
    return;
}


#line 26
__device__ void s_bwd_prop_read_t3_float3_0(uint idx_8, DiffTensorView_0 t3_2, float3  _s_dOut_6)
{
    uint2  _S126 = make_uint2 (idx_8, 0U);
    uint2  _S127 = make_uint2 (idx_8, 1U);

#line 28
    AtomicAdd_load_backward_0(t3_2.diff_1, make_uint2 (idx_8, 2U), _s_dOut_6.z);

#line 28
    AtomicAdd_load_backward_0(t3_2.diff_1, _S127, _s_dOut_6.y);

#line 28
    AtomicAdd_load_backward_0(t3_2.diff_1, _S126, _s_dOut_6.x);

#line 26
    return;
}


#line 245
__device__ void s_bwd_prop_load_splat_alphablend_0(int g_idx_2, DiffTensorView_0 xyz_vs_4, DiffTensorView_0 inv_cov_vs_4, DiffTensorView_0 opacity_4, DiffTensorView_0 rgb_4, DiffTensorView_0 distance_mu_4, DiffTensorView_0 distance_sigma_4, Splat_2D_AlphaBlend_0 _s_dOut_7, s_bwd_prop_load_splat_alphablend_Intermediates_0 _s_diff_ctx_1)
{

#line 253
    uint _S128 = uint(g_idx_2);

#line 261
    float3  _S129 = make_float3 (0.0f);

#line 261
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S130;

#line 261
    (&_S130)->primal_1 = _s_diff_ctx_1._S107;

#line 261
    (&_S130)->differential_0 = _S129;

#line 261
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S131;

#line 261
    (&_S131)->primal_1 = _s_diff_ctx_1._S108;

#line 261
    (&_S131)->differential_0 = _S129;

#line 261
    DiffPair_float_0 _S132;

#line 261
    (&_S132)->primal_1 = _s_diff_ctx_1._S109;

#line 261
    (&_S132)->differential_0 = 0.0f;

#line 261
    Matrix<float, 2, 2>  _S133 = makeMatrix<float, 2, 2> (0.0f);

#line 261
    DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 _S134;

#line 261
    (&_S134)->primal_1 = _s_diff_ctx_1._S110;

#line 261
    (&_S134)->differential_0 = _S133;

#line 261
    DiffPair_float_0 _S135;

#line 261
    (&_S135)->primal_1 = _s_diff_ctx_1._S111;

#line 261
    (&_S135)->differential_0 = 0.0f;

#line 261
    DiffPair_float_0 _S136;

#line 261
    (&_S136)->primal_1 = _s_diff_ctx_1._S112;

#line 261
    (&_S136)->differential_0 = 0.0f;

#line 261
    s_bwd_prop_Splat_2D_AlphaBlend_x24init_0(&_S130, &_S131, &_S132, &_S134, &_S135, &_S136, _s_dOut_7);

#line 261
    s_bwd_prop_read_t1_float_0(_S128, distance_sigma_4, _S136.differential_0);

#line 261
    s_bwd_prop_read_t1_float_0(_S128, distance_mu_4, _S135.differential_0);

#line 261
    s_bwd_prop_read_t2x2_float2x2_0(_S128, inv_cov_vs_4, _S134.differential_0);

#line 261
    s_bwd_prop_read_t1_float_0(_S128, opacity_4, _S132.differential_0);

#line 261
    s_bwd_prop_read_t3_float3_0(_S128, rgb_4, _S131.differential_0);

#line 261
    s_bwd_prop_read_t3_float3_0(_S128, xyz_vs_4, _S130.differential_0);

#line 245
    return;
}


#line 245
__device__ void s_bwd_load_splat_alphablend_0(int _S137, DiffTensorView_0 _S138, DiffTensorView_0 _S139, DiffTensorView_0 _S140, DiffTensorView_0 _S141, DiffTensorView_0 _S142, DiffTensorView_0 _S143, Splat_2D_AlphaBlend_0 _S144)
{

#line 251
    s_bwd_prop_load_splat_alphablend_Intermediates_0 _S145;

#line 251
    Splat_2D_AlphaBlend_0 _S146 = s_primal_ctx_load_splat_alphablend_0(_S137, _S138, _S139, _S140, _S141, _S142, _S143, &_S145);

#line 251
    s_bwd_prop_load_splat_alphablend_0(_S137, _S138, _S139, _S140, _S141, _S142, _S143, _S144, _S145);

#line 251
    return;
}


#line 117 "/data/fwindisch/hierarchical-LOD-gaussians/submodules/lod-slang-gaussian-rasterization/lod_slang_gaussian_rasterization/internal/slang/alphablend_shader.slang"
__device__ void bwd_alpha_blend_0(TensorView sorted_gauss_idx_0, DiffTensorView_0 xyz_vs_5, DiffTensorView_0 inv_cov_vs_5, DiffTensorView_0 opacity_5, DiffTensorView_0 rgb_5, DiffTensorView_0 distance_mu_5, DiffTensorView_0 distance_sigma_5, TensorView distance_1, DiffTensorView_0 final_pixel_state_0, TensorView n_contributors_0, uint2  pix_coord_1, uint tile_idx_start_0, uint tile_idx_end_0, uint tile_height_0, uint tile_width_0, uint H_2, uint W_2, float4  d_current_pixel_state_0)
{

#line 138
    uint _S147 = pix_coord_1.x;

#line 138
    bool is_inside_0;

#line 138
    if(_S147 < W_2)
    {

#line 138
        is_inside_0 = (pix_coord_1.y) < H_2;

#line 138
    }
    else
    {

#line 138
        is_inside_0 = false;

#line 138
    }
    uint block_size_0 = tile_height_0 * tile_width_0;
    uint _S148 = tile_idx_end_0 - tile_idx_start_0;

#line 140
    uint _S149 = (_S148 + block_size_0 - 1U) / block_size_0;

#line 140
    int _S150 = int(_S149);

    int _S151 = int(_S148);

#line 142
    int n_contrib_fwd_0;

#line 142
    float4  current_pixel_state_0;

#line 147
    if(is_inside_0)
    {

#line 148
        uint _S152 = pix_coord_1.y;

#line 148
        float4  _S153 = make_float4 (DiffTensorView_load_0(final_pixel_state_0, make_uint3 (_S152, _S147, 0U)), DiffTensorView_load_0(final_pixel_state_0, make_uint3 (_S152, _S147, 1U)), DiffTensorView_load_0(final_pixel_state_0, make_uint3 (_S152, _S147, 2U)), DiffTensorView_load_0(final_pixel_state_0, make_uint3 (_S152, _S147, 3U)));



        int _S154 = ((n_contributors_0).load<int>((_S152), (_S147), (0U)));

#line 152
        n_contrib_fwd_0 = _S154;

#line 152
        current_pixel_state_0 = _S153;

#line 147
    }

#line 155
    float2  _S155 = make_float2 ((float)pix_coord_1.x, (float)pix_coord_1.y);

    float2  _S156 = make_float2 (0.0f);

#line 157
    DiffPair_vectorx3Cfloatx2C2x3E_0 dp_center_pix_coord_0;

#line 157
    (&dp_center_pix_coord_0)->primal_1 = _S155;

#line 157
    (&dp_center_pix_coord_0)->differential_0 = _S156;


    uint3  _S157 = ((threadIdx));

#line 160
    uint _S158 = _S157.y * ((blockDim)).x + _S157.x;

#line 160
    float4  _S159 = d_current_pixel_state_0;

#line 160
    int i_5 = int(0);

#line 160
    int splats_left_to_process_0 = _S151;

#line 160
    uint current_splat_offset_0 = _S148;
    for(;;)
    {

#line 161
        if(i_5 < _S150)
        {
        }
        else
        {

#line 161
            break;
        }

        __syncthreads();

        uint _S160 = uint(int(uint(i_5) * block_size_0 + _S158));

#line 166
        if((tile_idx_start_0 + _S160) < tile_idx_end_0)
        {
            int _S161 = ((sorted_gauss_idx_0).load<int>((tile_idx_end_0 - _S160 - 1U)));

#line 168
            uint coll_id_0 = uint(_S161);
            (*&collected_idx_0)[_S158] = coll_id_0;
            (*&collected_splats_0)[_S158] = load_splat_alphablend_0(int(coll_id_0), xyz_vs_5, inv_cov_vs_5, opacity_5, rgb_5, distance_mu_5, distance_sigma_5);

#line 166
        }

#line 172
        __syncthreads();
        if(is_inside_0)
        {

#line 173
            float4  current_pixel_state_1 = current_pixel_state_0;

#line 173
            float4  _S162 = _S159;

#line 173
            int j_0 = int(0);

#line 173
            uint current_splat_offset_1 = current_splat_offset_0;
            for(;;)
            {

#line 174
                if(uint(j_0) < (U32_min((block_size_0), (uint(splats_left_to_process_0)))))
                {
                }
                else
                {

#line 174
                    break;
                }
                uint current_splat_offset_2 = current_splat_offset_1 - 1U;
                if(current_splat_offset_2 >= uint(n_contrib_fwd_0))
                {

#line 178
                    j_0 = j_0 + int(1);

#line 178
                    current_splat_offset_1 = current_splat_offset_2;

#line 174
                    continue;
                }



                uint g_idx_3 = (*&collected_idx_0)[j_0];
                Splat_2D_AlphaBlend_0 g_1 = (*&collected_splats_0)[j_0];

                float _S163 = ((distance_1).load<float>(((*&collected_idx_0)[j_0])));

#line 182
                float4  gauss_rgba_0 = evaluate_splat_0(g_1, _S155, _S163, H_2, W_2);

                if((gauss_rgba_0.w) < 0.00392156885936856f)
                {

#line 185
                    j_0 = j_0 + int(1);

#line 185
                    current_splat_offset_1 = current_splat_offset_2;

#line 174
                    continue;
                }

#line 190
                float4  current_pixel_state_2 = undo_pixel_state_0(current_pixel_state_1, gauss_rgba_0);

#line 196
                Splat_2D_AlphaBlend_0 _S164 = Splat_2D_AlphaBlend_x24_syn_dzero_0();

#line 196
                DiffPair_Splat_2D_AlphaBlend_0 dp_g_0;

#line 196
                (&dp_g_0)->primal_1 = g_1;

#line 196
                (&dp_g_0)->differential_0 = _S164;
                float _S165 = ((distance_1).load<float>((g_idx_3)));

#line 197
                DiffPair_float_0 dp_distance_0;

#line 197
                (&dp_distance_0)->primal_1 = _S165;

#line 197
                (&dp_distance_0)->differential_0 = 0.0f;
                float4  _S166 = make_float4 (0.0f);

#line 198
                DiffPair_vectorx3Cfloatx2C4x3E_0 dp_gauss_rgba_0;

#line 198
                (&dp_gauss_rgba_0)->primal_1 = gauss_rgba_0;

#line 198
                (&dp_gauss_rgba_0)->differential_0 = _S166;

                DiffPair_vectorx3Cfloatx2C4x3E_0 dp_current_pixel_state_0;

#line 200
                (&dp_current_pixel_state_0)->primal_1 = current_pixel_state_2;

#line 200
                (&dp_current_pixel_state_0)->differential_0 = _S166;

#line 205
                s_bwd_update_pixel_state_0(&dp_current_pixel_state_0, &dp_gauss_rgba_0, _S162);

#line 211
                s_bwd_evaluate_splat_0(&dp_g_0, &dp_center_pix_coord_0, &dp_distance_0, H_2, W_2, dp_gauss_rgba_0.differential_0);

                s_bwd_load_splat_alphablend_0(int(g_idx_3), xyz_vs_5, inv_cov_vs_5, opacity_5, rgb_5, distance_mu_5, distance_sigma_5, dp_g_0.differential_0);

#line 213
                current_pixel_state_1 = current_pixel_state_2;

#line 213
                _S162 = dp_current_pixel_state_0.differential_0;

#line 174
                j_0 = j_0 + int(1);

#line 174
                current_splat_offset_1 = current_splat_offset_2;

#line 174
            }

#line 174
            current_pixel_state_0 = current_pixel_state_1;

#line 174
            _S159 = _S162;

#line 174
            current_splat_offset_0 = current_splat_offset_1;

#line 173
        }

#line 216
        int splats_left_to_process_1 = splats_left_to_process_0 - int(block_size_0);

#line 161
        i_5 = i_5 + int(1);

#line 161
        splats_left_to_process_0 = splats_left_to_process_1;

#line 161
    }

#line 218
    return;
}


#line 43
__device__ float4  alpha_blend_0(TensorView sorted_gauss_idx_1, DiffTensorView_0 xyz_vs_6, DiffTensorView_0 inv_cov_vs_6, DiffTensorView_0 opacity_6, DiffTensorView_0 rgb_6, DiffTensorView_0 distance_mu_6, DiffTensorView_0 distance_sigma_6, TensorView distance_2, DiffTensorView_0 final_pixel_state_1, TensorView n_contributors_1, uint2  pix_coord_2, uint tile_idx_start_1, uint tile_idx_end_1, uint tile_height_1, uint tile_width_1, uint H_3, uint W_3)
{

#line 61
    float2  _S167 = make_float2 ((float)pix_coord_2.x, (float)pix_coord_2.y);
    float4  _S168 = make_float4 (0.0f, 0.0f, 0.0f, 1.0f);
    uint block_size_1 = tile_height_1 * tile_width_1;
    uint _S169 = pix_coord_2.x;

#line 64
    bool is_inside_1;

#line 64
    if(_S169 < W_3)
    {

#line 64
        is_inside_1 = (pix_coord_2.y) < H_3;

#line 64
    }
    else
    {

#line 64
        is_inside_1 = false;

#line 64
    }

    uint _S170 = tile_idx_end_1 - tile_idx_start_1;

#line 66
    uint _S171 = (_S170 + block_size_1 - 1U) / block_size_1;

#line 66
    int _S172 = int(_S171);
    uint3  _S173 = ((threadIdx));

#line 67
    uint _S174 = _S173.y * ((blockDim)).x + _S173.x;

    int _S175 = int(_S170);

#line 69
    bool thread_active_0 = is_inside_1;

#line 69
    float4  curr_pixel_state_0 = _S168;

#line 69
    int i_6 = int(0);

#line 69
    int splats_left_to_process_2 = _S175;

#line 69
    int local_n_contrib_0 = int(0);
    for(;;)
    {

#line 70
        if(i_6 < _S172)
        {
        }
        else
        {

#line 70
            break;
        }

        __syncthreads();

        uint _S176 = tile_idx_start_1 + uint(int(uint(i_6) * block_size_1 + _S174));

#line 75
        if(_S176 < tile_idx_end_1)
        {
            int _S177 = ((sorted_gauss_idx_1).load<int>((_S176)));

#line 77
            uint coll_id_1 = uint(_S177);
            (*&collected_idx_0)[_S174] = coll_id_1;
            (*&collected_splats_0)[_S174] = load_splat_alphablend_0(int(coll_id_1), xyz_vs_6, inv_cov_vs_6, opacity_6, rgb_6, distance_mu_6, distance_sigma_6);

#line 75
        }

#line 81
        __syncthreads();

#line 81
        float4  curr_pixel_state_1;
        if(thread_active_0)
        {

#line 82
            int local_n_contrib_1;

#line 82
            bool thread_active_1;

#line 82
            curr_pixel_state_1 = curr_pixel_state_0;

#line 82
            int j_1 = int(0);

#line 82
            int local_n_contrib_2 = local_n_contrib_0;
            for(;;)
            {

#line 83
                if(uint(j_1) < (U32_min((block_size_1), (uint(splats_left_to_process_2)))))
                {
                }
                else
                {

#line 83
                    thread_active_1 = thread_active_0;

#line 83
                    local_n_contrib_1 = local_n_contrib_2;

#line 83
                    break;
                }
                int local_n_contrib_3 = local_n_contrib_2 + int(1);
                Splat_2D_AlphaBlend_0 g_2 = (*&collected_splats_0)[j_1];
                float _S178 = ((distance_2).load<float>(((*&collected_idx_0)[j_1])));

#line 87
                float4  gauss_rgba_1 = evaluate_splat_0(g_2, _S167, _S178, H_3, W_3);


                if((gauss_rgba_1.w) < 0.00392156885936856f)
                {

#line 91
                    j_1 = j_1 + int(1);

#line 91
                    local_n_contrib_2 = local_n_contrib_3;

#line 83
                    continue;
                }

#line 93
                float4  new_pixel_state_0 = update_pixel_state_0(curr_pixel_state_1, gauss_rgba_1);


                if((new_pixel_state_0.w) < 0.00009999999747379f)
                {
                    int _S179 = local_n_contrib_3 - int(1);

#line 98
                    thread_active_1 = false;

#line 98
                    local_n_contrib_1 = _S179;

                    break;
                }

#line 100
                curr_pixel_state_1 = new_pixel_state_0;

#line 83
                j_1 = j_1 + int(1);

#line 83
                local_n_contrib_2 = local_n_contrib_3;

#line 83
            }

#line 83
            thread_active_0 = thread_active_1;

#line 83
            local_n_contrib_0 = local_n_contrib_1;

#line 82
        }
        else
        {

#line 82
            curr_pixel_state_1 = curr_pixel_state_0;

#line 82
        }

#line 105
        int splats_left_to_process_3 = splats_left_to_process_2 - int(block_size_1);

#line 70
        int _S180 = i_6 + int(1);

#line 70
        curr_pixel_state_0 = curr_pixel_state_1;

#line 70
        i_6 = _S180;

#line 70
        splats_left_to_process_2 = splats_left_to_process_3;

#line 70
    }

#line 108
    if(is_inside_1)
    {

#line 109
        (n_contributors_1).store<int>((pix_coord_2.y), (_S169), (0U), (local_n_contrib_0));

#line 108
    }

    return curr_pixel_state_0;
}


#line 939 "diff.meta.slang"
__device__ void AtomicAdd_storeOnce_forward_0(AtomicAdd_0 this_5, uint3  i_7, float dx_0)
{
    (this_5.diff_0).store<float>((i_7), (dx_0));
    return;
}


#line 1184
__device__ void DiffTensorView_storeOnce_forward_0(DiffTensorView_0 this_6, uint3  x_0, DiffPair_float_0 dpval_0)
{
    (this_6.primal_0).store<float>((x_0), (dpval_0.primal_1));
    AtomicAdd_storeOnce_forward_0(this_6.diff_1, x_0, dpval_0.differential_0);
    return;
}


#line 1175
__device__ void DiffTensorView_storeOnce_0(DiffTensorView_0 this_7, uint3  x_1, float val_0)
{

#line 1175
    (this_7.primal_0).store<float>((x_1), (val_0));

#line 1175
    return;
}


#line 246 "/data/fwindisch/hierarchical-LOD-gaussians/submodules/lod-slang-gaussian-rasterization/lod_slang_gaussian_rasterization/internal/slang/alphablend_shader.slang"
struct s_bwd_prop_splat_tiled_Intermediates_0
{
    int _S181;
    int _S182;
};


#line 246
__device__ float4  s_primal_ctx_alpha_blend_0(TensorView _S183, DiffTensorView_0 _S184, DiffTensorView_0 _S185, DiffTensorView_0 _S186, DiffTensorView_0 _S187, DiffTensorView_0 _S188, DiffTensorView_0 _S189, TensorView _S190, DiffTensorView_0 _S191, TensorView _S192, uint2  _S193, uint _S194, uint _S195, uint _S196, uint _S197, uint _S198, uint _S199)
{

#line 246
    float4  _S200 = alpha_blend_0(_S183, _S184, _S185, _S186, _S187, _S188, _S189, _S190, _S191, _S192, _S193, _S194, _S195, _S196, _S197, _S198, _S199);

#line 246
    return _S200;
}


#line 246
__device__ void s_primal_ctx_splat_tiled_0(TensorView sorted_gauss_idx_2, TensorView tile_ranges_0, DiffTensorView_0 xyz_vs_7, DiffTensorView_0 inv_cov_vs_7, DiffTensorView_0 opacity_7, DiffTensorView_0 rgb_7, DiffTensorView_0 distance_mu_7, DiffTensorView_0 distance_sigma_7, TensorView distance_3, DiffTensorView_0 output_img_0, TensorView n_contributors_2, int grid_height_0, int grid_width_0, int tile_height_2, int tile_width_2, s_bwd_prop_splat_tiled_Intermediates_0 * _s_diff_ctx_2)
{

#line 260
    _s_diff_ctx_2->_S181 = int(0);

#line 260
    _s_diff_ctx_2->_S182 = int(0);

#line 267
    _s_diff_ctx_2->_S181 = int(0);
    _s_diff_ctx_2->_S182 = int(0);

#line 262
    uint3  _S201 = ((blockIdx));

    uint2  pix_coord_3 = uint2 {(_S201 * ((blockDim)) + ((threadIdx))).x, (_S201 * ((blockDim)) + ((threadIdx))).y};

    uint tile_idx_0 = _S201.y * uint(grid_width_0) + _S201.x;
    int _S202 = ((tile_ranges_0).load<int>((tile_idx_0), (0U)));

#line 267
    _s_diff_ctx_2->_S181 = _S202;

#line 267
    uint tile_idx_start_2 = uint(_S202);
    int _S203 = ((tile_ranges_0).load<int>((tile_idx_0), (1U)));

#line 268
    _s_diff_ctx_2->_S182 = _S203;

#line 268
    uint tile_idx_end_2 = uint(_S203);

    uint _S204 = pix_coord_3.x;

#line 270
    uint _S205 = DiffTensorView_size_0(output_img_0, 1U);

#line 270
    bool is_inside_2;

#line 270
    if(_S204 < _S205)
    {

#line 270
        is_inside_2 = (pix_coord_3.y) < (DiffTensorView_size_0(output_img_0, 0U));

#line 270
    }
    else
    {

#line 270
        is_inside_2 = false;

#line 270
    }

#line 270
    float4  _S206 = s_primal_ctx_alpha_blend_0(sorted_gauss_idx_2, xyz_vs_7, inv_cov_vs_7, opacity_7, rgb_7, distance_mu_7, distance_sigma_7, distance_3, output_img_0, n_contributors_2, pix_coord_3, tile_idx_start_2, tile_idx_end_2, uint(tile_height_2), uint(tile_width_2), DiffTensorView_size_0(output_img_0, 0U), _S205);

#line 290
    if(is_inside_2)
    {

#line 291
        uint _S207 = pix_coord_3.y;

#line 291
        DiffTensorView_storeOnce_0(output_img_0, make_uint3 (_S207, _S204, 0U), _S206.x);
        DiffTensorView_storeOnce_0(output_img_0, make_uint3 (_S207, _S204, 1U), _S206.y);
        DiffTensorView_storeOnce_0(output_img_0, make_uint3 (_S207, _S204, 2U), _S206.z);
        DiffTensorView_storeOnce_0(output_img_0, make_uint3 (_S207, _S204, 3U), _S206.w);

#line 290
    }

#line 290
    return;
}


#line 951 "diff.meta.slang"
__device__ float AtomicAdd_storeOnce_backward_0(AtomicAdd_0 this_8, uint3  i_8)
{
    float _S208 = ((this_8.diff_0).load<float>((i_8)));

#line 953
    return _S208;
}


#line 953
__device__ void s_bwd_prop_alpha_blend_0(TensorView _S209, DiffTensorView_0 _S210, DiffTensorView_0 _S211, DiffTensorView_0 _S212, DiffTensorView_0 _S213, DiffTensorView_0 _S214, DiffTensorView_0 _S215, TensorView _S216, DiffTensorView_0 _S217, TensorView _S218, uint2  _S219, uint _S220, uint _S221, uint _S222, uint _S223, uint _S224, uint _S225, float4  _S226)
{

#line 953
    bwd_alpha_blend_0(_S209, _S210, _S211, _S212, _S213, _S214, _S215, _S216, _S217, _S218, _S219, _S220, _S221, _S222, _S223, _S224, _S225, _S226);

#line 953
    return;
}


#line 246 "/data/fwindisch/hierarchical-LOD-gaussians/submodules/lod-slang-gaussian-rasterization/lod_slang_gaussian_rasterization/internal/slang/alphablend_shader.slang"
__device__ void s_bwd_prop_splat_tiled_0(TensorView sorted_gauss_idx_3, TensorView tile_ranges_1, DiffTensorView_0 xyz_vs_8, DiffTensorView_0 inv_cov_vs_8, DiffTensorView_0 opacity_8, DiffTensorView_0 rgb_8, DiffTensorView_0 distance_mu_8, DiffTensorView_0 distance_sigma_8, TensorView distance_4, DiffTensorView_0 output_img_1, TensorView n_contributors_3, int grid_height_1, int grid_width_1, int tile_height_3, int tile_width_3, s_bwd_prop_splat_tiled_Intermediates_0 _s_diff_ctx_3)
{

#line 291
    uint3  _S227 = make_uint3 (0U);

#line 264
    uint2  pix_coord_4 = uint2 {(((blockIdx)) * ((blockDim)) + ((threadIdx))).x, (((blockIdx)) * ((blockDim)) + ((threadIdx))).y};


    uint tile_idx_start_3 = uint(_s_diff_ctx_3._S181);
    uint tile_idx_end_3 = uint(_s_diff_ctx_3._S182);

    uint _S228 = pix_coord_4.x;

#line 270
    uint _S229 = DiffTensorView_size_0(output_img_1, 1U);

#line 270
    bool is_inside_3;

#line 270
    if(_S228 < _S229)
    {

#line 270
        is_inside_3 = (pix_coord_4.y) < (DiffTensorView_size_0(output_img_1, 0U));

#line 270
    }
    else
    {

#line 270
        is_inside_3 = false;

#line 270
    }

#line 284
    uint _S230 = uint(tile_height_3);
    uint _S231 = uint(tile_width_3);
    uint _S232 = DiffTensorView_size_0(output_img_1, 0U);

#line 286
    uint3  _S233;

#line 286
    uint3  _S234;

#line 286
    uint3  _S235;

#line 286
    uint3  _S236;



    if(is_inside_3)
    {

#line 291
        uint _S237 = pix_coord_4.y;

#line 291
        uint3  _S238 = make_uint3 (_S237, _S228, 0U);
        uint3  _S239 = make_uint3 (_S237, _S228, 1U);
        uint3  _S240 = make_uint3 (_S237, _S228, 2U);

#line 293
        _S233 = make_uint3 (_S237, _S228, 3U);

#line 293
        _S234 = _S240;

#line 293
        _S235 = _S239;

#line 293
        _S236 = _S238;

#line 293
    }
    else
    {

#line 293
        _S233 = _S227;

#line 293
        _S234 = _S227;

#line 293
        _S235 = _S227;

#line 293
        _S236 = _S227;

#line 293
    }

#line 271
    float4  _S241 = make_float4 (0.0f);

#line 271
    float4  _S242;

#line 271
    if(is_inside_3)
    {

#line 271
        _S242 = make_float4 (AtomicAdd_storeOnce_backward_0(output_img_1.diff_1, _S236), AtomicAdd_storeOnce_backward_0(output_img_1.diff_1, _S235), AtomicAdd_storeOnce_backward_0(output_img_1.diff_1, _S234), AtomicAdd_storeOnce_backward_0(output_img_1.diff_1, _S233));

#line 271
    }
    else
    {

#line 271
        _S242 = _S241;

#line 271
    }

#line 271
    s_bwd_prop_alpha_blend_0(sorted_gauss_idx_3, xyz_vs_8, inv_cov_vs_8, opacity_8, rgb_8, distance_mu_8, distance_sigma_8, distance_4, output_img_1, n_contributors_3, pix_coord_4, tile_idx_start_3, tile_idx_end_3, _S230, _S231, _S232, _S229, _S242);

#line 246
    return;
}


#line 246
__device__ void s_bwd_splat_tiled_0(TensorView _S243, TensorView _S244, DiffTensorView_0 _S245, DiffTensorView_0 _S246, DiffTensorView_0 _S247, DiffTensorView_0 _S248, DiffTensorView_0 _S249, DiffTensorView_0 _S250, TensorView _S251, DiffTensorView_0 _S252, TensorView _S253, int _S254, int _S255, int _S256, int _S257)
{

#line 260
    s_bwd_prop_splat_tiled_Intermediates_0 _S258;

#line 260
    s_primal_ctx_splat_tiled_0(_S243, _S244, _S245, _S246, _S247, _S248, _S249, _S250, _S251, _S252, _S253, _S254, _S255, _S256, _S257, &_S258);

#line 260
    s_bwd_prop_splat_tiled_0(_S243, _S244, _S245, _S246, _S247, _S248, _S249, _S250, _S251, _S252, _S253, _S254, _S255, _S256, _S257, _S258);

#line 260
    return;
}


#line 260
extern "C" {
__global__ void __kernel__splat_tiled_bwd_diff(TensorView sorted_gauss_idx_4, TensorView tile_ranges_2, DiffTensorView_0 xyz_vs_9, DiffTensorView_0 inv_cov_vs_9, DiffTensorView_0 opacity_9, DiffTensorView_0 rgb_9, DiffTensorView_0 distance_mu_9, DiffTensorView_0 distance_sigma_9, TensorView distance_5, DiffTensorView_0 output_img_2, TensorView n_contributors_4, int grid_height_2, int grid_width_2, int tile_height_4, int tile_width_4)
{

#line 260
    s_bwd_splat_tiled_0(sorted_gauss_idx_4, tile_ranges_2, xyz_vs_9, inv_cov_vs_9, opacity_9, rgb_9, distance_mu_9, distance_sigma_9, distance_5, output_img_2, n_contributors_4, grid_height_2, grid_width_2, tile_height_4, tile_width_4);

#line 260
    return;
}

}

#line 856 "diff.meta.slang"
__device__ float AtomicAdd_load_forward_0(AtomicAdd_0 this_9, uint2  i_9)
{
    float _S259 = ((this_9.diff_0).load<float>((i_9)));

#line 858
    return _S259;
}


#line 858
__device__ DiffPair_vectorx3Cfloatx2C3x3E_0 s_fwd_read_t3_float3_0(uint idx_9, DiffTensorView_0 t3_3)
{

#line 28 "/data/fwindisch/hierarchical-LOD-gaussians/submodules/lod-slang-gaussian-rasterization/lod_slang_gaussian_rasterization/internal/slang/utils.slang"
    uint2  _S260 = make_uint2 (idx_9, 0U);

#line 28
    float _S261 = ((t3_3.primal_0).load<float>((_S260)));

#line 28
    float _S262 = AtomicAdd_load_forward_0(t3_3.diff_1, _S260);
    uint2  _S263 = make_uint2 (idx_9, 1U);

#line 28
    float _S264 = ((t3_3.primal_0).load<float>((_S263)));

#line 28
    float _S265 = AtomicAdd_load_forward_0(t3_3.diff_1, _S263);

    uint2  _S266 = make_uint2 (idx_9, 2U);

#line 28
    float _S267 = ((t3_3.primal_0).load<float>((_S266)));

#line 28
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S268 = { make_float3 (_S261, _S264, _S267), make_float3 (_S262, _S265, AtomicAdd_load_forward_0(t3_3.diff_1, _S266)) };

#line 28
    return _S268;
}


#line 255
__device__ DiffPair_float_0 s_fwd_read_t1_float_0(uint idx_10, DiffTensorView_0 t1_3)
{

#line 22
    uint2  _S269 = make_uint2 (idx_10, 0U);

#line 22
    float _S270 = ((t1_3.primal_0).load<float>((_S269)));

#line 22
    DiffPair_float_0 _S271 = { _S270, AtomicAdd_load_forward_0(t1_3.diff_1, _S269) };

#line 22
    return _S271;
}


#line 856 "diff.meta.slang"
__device__ float AtomicAdd_load_forward_1(AtomicAdd_0 this_10, uint3  i_10)
{
    float _S272 = ((this_10.diff_0).load<float>((i_10)));

#line 858
    return _S272;
}


#line 858
__device__ DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 s_fwd_read_t2x2_float2x2_0(uint idx_11, DiffTensorView_0 t2x2_3)
{

#line 54 "/data/fwindisch/hierarchical-LOD-gaussians/submodules/lod-slang-gaussian-rasterization/lod_slang_gaussian_rasterization/internal/slang/utils.slang"
    uint3  _S273 = make_uint3 (idx_11, 0U, 0U);

#line 54
    float _S274 = ((t2x2_3.primal_0).load<float>((_S273)));

#line 54
    float _S275 = AtomicAdd_load_forward_1(t2x2_3.diff_1, _S273);
    uint3  _S276 = make_uint3 (idx_11, 1U, 0U);

#line 54
    float _S277 = ((t2x2_3.primal_0).load<float>((_S276)));

#line 54
    float _S278 = AtomicAdd_load_forward_1(t2x2_3.diff_1, _S276);

    uint3  _S279 = make_uint3 (idx_11, 0U, 1U);

#line 54
    float _S280 = ((t2x2_3.primal_0).load<float>((_S279)));

#line 54
    float _S281 = AtomicAdd_load_forward_1(t2x2_3.diff_1, _S279);


    uint3  _S282 = make_uint3 (idx_11, 1U, 1U);

#line 54
    float _S283 = ((t2x2_3.primal_0).load<float>((_S282)));

#line 54
    DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 _S284 = { makeMatrix<float, 2, 2> (_S274, _S277, _S280, _S283), makeMatrix<float, 2, 2> (_S275, _S278, _S281, AtomicAdd_load_forward_1(t2x2_3.diff_1, _S282)) };

#line 54
    return _S284;
}


#line 261
__device__ DiffPair_Splat_2D_AlphaBlend_0 s_fwd_Splat_2D_AlphaBlend_x24init_0(DiffPair_vectorx3Cfloatx2C3x3E_0 dpxyz_vs_2, DiffPair_vectorx3Cfloatx2C3x3E_0 dprgb_2, DiffPair_float_0 dpopacity_2, DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 dpinv_cov_vs_2, DiffPair_float_0 dpdistance_mu_2, DiffPair_float_0 dpdistance_sigma_2)
{

#line 241
    Splat_2D_AlphaBlend_0 _S285 = { dpxyz_vs_2.primal_1, dprgb_2.primal_1, dpopacity_2.primal_1, dpinv_cov_vs_2.primal_1, dpdistance_mu_2.primal_1, dpdistance_sigma_2.primal_1 };

#line 241
    Splat_2D_AlphaBlend_0 _S286 = { dpxyz_vs_2.differential_0, dprgb_2.differential_0, dpopacity_2.differential_0, dpinv_cov_vs_2.differential_0, dpdistance_mu_2.differential_0, dpdistance_sigma_2.differential_0 };

#line 241
    DiffPair_Splat_2D_AlphaBlend_0 _S287 = { _S285, _S286 };

#line 234
    return _S287;
}


#line 234
__device__ DiffPair_Splat_2D_AlphaBlend_0 s_fwd_load_splat_alphablend_0(int g_idx_4, DiffTensorView_0 xyz_vs_10, DiffTensorView_0 inv_cov_vs_10, DiffTensorView_0 opacity_10, DiffTensorView_0 rgb_10, DiffTensorView_0 distance_mu_10, DiffTensorView_0 distance_sigma_10)
{

#line 253
    uint _S288 = uint(g_idx_4);

#line 253
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S289 = s_fwd_read_t3_float3_0(_S288, xyz_vs_10);
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S290 = s_fwd_read_t3_float3_0(_S288, rgb_10);
    DiffPair_float_0 _S291 = s_fwd_read_t1_float_0(_S288, opacity_10);
    DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 _S292 = s_fwd_read_t2x2_float2x2_0(_S288, inv_cov_vs_10);

    DiffPair_float_0 _S293 = s_fwd_read_t1_float_0(_S288, distance_mu_10);
    DiffPair_float_0 _S294 = s_fwd_read_t1_float_0(_S288, distance_sigma_10);

#line 259
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S295 = { _S289.primal_1, _S289.differential_0 };

#line 259
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S296 = { _S290.primal_1, _S290.differential_0 };

#line 259
    DiffPair_float_0 _S297 = { _S291.primal_1, _S291.differential_0 };

#line 259
    DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 _S298 = { _S292.primal_1, _S292.differential_0 };

#line 259
    DiffPair_float_0 _S299 = { _S293.primal_1, _S293.differential_0 };

#line 259
    DiffPair_float_0 _S300 = { _S294.primal_1, _S294.differential_0 };

    DiffPair_Splat_2D_AlphaBlend_0 _S301 = s_fwd_Splat_2D_AlphaBlend_x24init_0(_S295, _S296, _S297, _S298, _S299, _S300);

#line 261
    DiffPair_Splat_2D_AlphaBlend_0 _S302 = { _S301.primal_1, _S301.differential_0 };

#line 261
    return _S302;
}


#line 301
__device__ DiffPair_float_0 s_fwd_ndc2pix_0(DiffPair_float_0 dpv_2, int S_3)
{

#line 63
    float _S303 = float(S_3);

#line 63
    DiffPair_float_0 _S304 = { ((dpv_2.primal_1 + 1.0f) * _S303 - 1.0f) * 0.5f, dpv_2.differential_0 * _S303 * 0.5f };

#line 63
    return _S304;
}


#line 63
__device__ DiffPair_vectorx3Cfloatx2C4x3E_0 s_fwd_evaluate_splat_0(DiffPair_Splat_2D_AlphaBlend_0 dpg_1, DiffPair_vectorx3Cfloatx2C2x3E_0 dppix_coord_1, DiffPair_float_0 dpdistance_1, uint H_4, uint W_4)
{

#line 293
    DiffPair_float_0 _S305 = { dpg_1.primal_1.xyz_vs_0.x, dpg_1.differential_0.xyz_vs_0.x };

#line 301
    DiffPair_float_0 _S306 = s_fwd_ndc2pix_0(_S305, int(W_4));

#line 301
    float _S307 = dppix_coord_1.primal_1.x - _S306.primal_1;

#line 301
    float _S308 = dppix_coord_1.differential_0.x - _S306.differential_0;

#line 301
    DiffPair_float_0 _S309 = { dpg_1.primal_1.xyz_vs_0.y, dpg_1.differential_0.xyz_vs_0.y };
    DiffPair_float_0 _S310 = s_fwd_ndc2pix_0(_S309, int(H_4));

#line 302
    float _S311 = dppix_coord_1.primal_1.y - _S310.primal_1;

#line 302
    float _S312 = dppix_coord_1.differential_0.y - _S310.differential_0;
    float _S313 = dpg_1.primal_1.inv_cov_vs_0.rows[int(0)].x * _S307;
    float _S314 = dpg_1.primal_1.inv_cov_vs_0.rows[int(1)].y * _S311;

#line 304
    float _S315 = dpg_1.primal_1.inv_cov_vs_0.rows[int(0)].y + dpg_1.primal_1.inv_cov_vs_0.rows[int(1)].x;

#line 304
    float _S316 = _S315 * _S307;

#line 304
    DiffPair_float_0 _S317 = { -0.5f * (_S313 * _S307 + _S314 * _S311 + _S316 * _S311), ((dpg_1.differential_0.inv_cov_vs_0.rows[int(0)].x * _S307 + _S308 * dpg_1.primal_1.inv_cov_vs_0.rows[int(0)].x) * _S307 + _S308 * _S313 + ((dpg_1.differential_0.inv_cov_vs_0.rows[int(1)].y * _S311 + _S312 * dpg_1.primal_1.inv_cov_vs_0.rows[int(1)].y) * _S311 + _S312 * _S314) + (((dpg_1.differential_0.inv_cov_vs_0.rows[int(0)].y + dpg_1.differential_0.inv_cov_vs_0.rows[int(1)].x) * _S307 + _S308 * _S315) * _S311 + _S312 * _S316)) * -0.5f };
    DiffPair_float_0 _S318 = _d_exp_1(_S317);

#line 305
    DiffPair_float_0 _S319 = { 0.99000000953674316f, 0.0f };

#line 305
    DiffPair_float_0 _S320 = { dpg_1.primal_1.opacity_0 * _S318.primal_1, dpg_1.differential_0.opacity_0 * _S318.primal_1 + _S318.differential_0 * dpg_1.primal_1.opacity_0 };

#line 305
    DiffPair_float_0 _S321 = _d_min_1(_S319, _S320);
    float _S322 = dpdistance_1.primal_1 - dpg_1.primal_1.distance_mu_0;

#line 306
    float _S323 = (dpdistance_1.differential_0 - dpg_1.differential_0.distance_mu_0) * _S322;

#line 306
    float _S324 = -0.5f * (_S322 * _S322);

#line 306
    float _S325 = dpg_1.primal_1.distance_sigma_0;

#line 306
    float _S326 = dpg_1.differential_0.distance_sigma_0 * dpg_1.primal_1.distance_sigma_0;

#line 306
    float _S327 = _S325 * _S325 + 1.00000001168609742e-07f;

#line 306
    DiffPair_float_0 _S328 = { _S324 / _S327, ((_S323 + _S323) * -0.5f * _S327 - _S324 * (_S326 + _S326)) / (_S327 * _S327) };

#line 306
    DiffPair_float_0 _S329 = _d_exp_1(_S328);

#line 306
    float alpha_1 = _S321.primal_1 * _S329.primal_1;

#line 306
    float s_diff_alpha_0 = _S321.differential_0 * _S329.primal_1 + _S329.differential_0 * _S321.primal_1;

#line 306
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S330 = { make_float4 ((dpg_1.primal_1.rgb_0 * make_float3 (alpha_1)).x, (dpg_1.primal_1.rgb_0 * make_float3 (alpha_1)).y, (dpg_1.primal_1.rgb_0 * make_float3 (alpha_1)).z, alpha_1), make_float4 ((dpg_1.differential_0.rgb_0 * make_float3 (alpha_1) + make_float3 (s_diff_alpha_0) * dpg_1.primal_1.rgb_0).x, (dpg_1.differential_0.rgb_0 * make_float3 (alpha_1) + make_float3 (s_diff_alpha_0) * dpg_1.primal_1.rgb_0).y, (dpg_1.differential_0.rgb_0 * make_float3 (alpha_1) + make_float3 (s_diff_alpha_0) * dpg_1.primal_1.rgb_0).z, s_diff_alpha_0) };


    return _S330;
}


#line 93 "/data/fwindisch/hierarchical-LOD-gaussians/submodules/lod-slang-gaussian-rasterization/lod_slang_gaussian_rasterization/internal/slang/alphablend_shader.slang"
__device__ DiffPair_vectorx3Cfloatx2C4x3E_0 s_fwd_update_pixel_state_0(DiffPair_vectorx3Cfloatx2C4x3E_0 dppixel_state_t_nm1_1, DiffPair_vectorx3Cfloatx2C4x3E_0 dpgauss_rgba_t_n_1)
{

#line 29
    float3  _S331 = float3 {dpgauss_rgba_t_n_1.primal_1.x, dpgauss_rgba_t_n_1.primal_1.y, dpgauss_rgba_t_n_1.primal_1.z};

#line 29
    float _S332 = dppixel_state_t_nm1_1.primal_1.w;

#line 29
    float _S333 = dppixel_state_t_nm1_1.differential_0.w;
    float _S334 = 1.0f - dpgauss_rgba_t_n_1.primal_1.w;

#line 30
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S335 = { make_float4 ((float3 {dppixel_state_t_nm1_1.primal_1.x, dppixel_state_t_nm1_1.primal_1.y, dppixel_state_t_nm1_1.primal_1.z} + _S331 * make_float3 (_S332)).x, (float3 {dppixel_state_t_nm1_1.primal_1.x, dppixel_state_t_nm1_1.primal_1.y, dppixel_state_t_nm1_1.primal_1.z} + _S331 * make_float3 (_S332)).y, (float3 {dppixel_state_t_nm1_1.primal_1.x, dppixel_state_t_nm1_1.primal_1.y, dppixel_state_t_nm1_1.primal_1.z} + _S331 * make_float3 (_S332)).z, _S332 * _S334), make_float4 ((float3 {dppixel_state_t_nm1_1.differential_0.x, dppixel_state_t_nm1_1.differential_0.y, dppixel_state_t_nm1_1.differential_0.z} + (float3 {dpgauss_rgba_t_n_1.differential_0.x, dpgauss_rgba_t_n_1.differential_0.y, dpgauss_rgba_t_n_1.differential_0.z} * make_float3 (_S332) + make_float3 (_S333) * _S331)).x, (float3 {dppixel_state_t_nm1_1.differential_0.x, dppixel_state_t_nm1_1.differential_0.y, dppixel_state_t_nm1_1.differential_0.z} + (float3 {dpgauss_rgba_t_n_1.differential_0.x, dpgauss_rgba_t_n_1.differential_0.y, dpgauss_rgba_t_n_1.differential_0.z} * make_float3 (_S332) + make_float3 (_S333) * _S331)).y, (float3 {dppixel_state_t_nm1_1.differential_0.x, dppixel_state_t_nm1_1.differential_0.y, dppixel_state_t_nm1_1.differential_0.z} + (float3 {dpgauss_rgba_t_n_1.differential_0.x, dpgauss_rgba_t_n_1.differential_0.y, dpgauss_rgba_t_n_1.differential_0.z} * make_float3 (_S332) + make_float3 (_S333) * _S331)).z, _S333 * _S334 + (0.0f - dpgauss_rgba_t_n_1.differential_0.w) * _S332) };
    return _S335;
}


#line 31
__device__ DiffPair_vectorx3Cfloatx2C4x3E_0 s_fwd_alpha_blend_0(TensorView sorted_gauss_idx_5, DiffTensorView_0 xyz_vs_11, DiffTensorView_0 inv_cov_vs_11, DiffTensorView_0 opacity_11, DiffTensorView_0 rgb_11, DiffTensorView_0 distance_mu_11, DiffTensorView_0 distance_sigma_11, TensorView distance_6, DiffTensorView_0 final_pixel_state_2, TensorView n_contributors_5, uint2  pix_coord_5, uint tile_idx_start_4, uint tile_idx_end_4, uint tile_height_5, uint tile_width_5, uint H_5, uint W_5)
{

#line 61
    float2  _S336 = make_float2 ((float)pix_coord_5.x, (float)pix_coord_5.y);
    float4  _S337 = make_float4 (0.0f, 0.0f, 0.0f, 1.0f);

#line 62
    float4  _S338 = make_float4 (0.0f, 0.0f, 0.0f, 0.0f);
    uint block_size_2 = tile_height_5 * tile_width_5;
    uint _S339 = pix_coord_5.x;

#line 64
    bool is_inside_4;

#line 64
    if(_S339 < W_5)
    {

#line 64
        is_inside_4 = (pix_coord_5.y) < H_5;

#line 64
    }
    else
    {

#line 64
        is_inside_4 = false;

#line 64
    }

    uint _S340 = tile_idx_end_4 - tile_idx_start_4;

#line 66
    uint _S341 = (_S340 + block_size_2 - 1U) / block_size_2;

#line 66
    int _S342 = int(_S341);
    uint3  _S343 = ((threadIdx));

#line 67
    uint _S344 = _S343.y * ((blockDim)).x + _S343.x;

    int _S345 = int(_S340);

#line 105
    int _S346 = int(block_size_2);

#line 105
    bool thread_active_2 = is_inside_4;

#line 105
    float4  curr_pixel_state_2 = _S337;

#line 105
    float4  s_diff_curr_pixel_state_0 = _S338;

#line 105
    int i_11 = int(0);

#line 105
    int splats_left_to_process_4 = _S345;

#line 105
    int local_n_contrib_4 = int(0);

#line 70
    for(;;)
    {

#line 70
        if(i_11 < _S342)
        {
        }
        else
        {

#line 70
            break;
        }

        __syncthreads();

        uint _S347 = tile_idx_start_4 + uint(int(uint(i_11) * block_size_2 + _S344));

#line 75
        if(_S347 < tile_idx_end_4)
        {
            int _S348 = ((sorted_gauss_idx_5).load<int>((_S347)));

#line 77
            uint coll_id_2 = uint(_S348);

#line 77
            FixedArray<uint, 64>  _S349 = *&collected_idx_0;

#line 77
            _S349[_S344] = coll_id_2;
            *&collected_idx_0 = _S349;
            DiffPair_Splat_2D_AlphaBlend_0 _S350 = s_fwd_load_splat_alphablend_0(int(coll_id_2), xyz_vs_11, inv_cov_vs_11, opacity_11, rgb_11, distance_mu_11, distance_sigma_11);

#line 79
            FixedArray<Splat_2D_AlphaBlend_0, 64>  _S351 = *&collected_splats_0;

#line 79
            _S351[_S344] = _S350.primal_1;

#line 79
            *&collected_splats_0 = _S351;

#line 75
        }

#line 81
        __syncthreads();

#line 81
        float4  curr_pixel_state_3;

#line 81
        float4  s_diff_curr_pixel_state_1;
        if(thread_active_2)
        {

#line 82
            int local_n_contrib_5;

#line 82
            bool thread_active_3;
            uint _S352 = (U32_min((block_size_2), (uint(splats_left_to_process_4))));

#line 83
            curr_pixel_state_3 = curr_pixel_state_2;

#line 83
            s_diff_curr_pixel_state_1 = s_diff_curr_pixel_state_0;

#line 83
            int j_2 = int(0);

#line 83
            int local_n_contrib_6 = local_n_contrib_4;

#line 83
            for(;;)
            {

#line 83
                if(uint(j_2) < _S352)
                {
                }
                else
                {

#line 83
                    thread_active_3 = thread_active_2;

#line 83
                    local_n_contrib_5 = local_n_contrib_6;

#line 83
                    break;
                }
                int local_n_contrib_7 = local_n_contrib_6 + int(1);
                FixedArray<Splat_2D_AlphaBlend_0, 64>  _S353 = *&collected_splats_0;

#line 83
                int _S354 = j_2;



                float _S355 = ((distance_6).load<float>(((*&collected_idx_0)[j_2])));

#line 87
                DiffPair_Splat_2D_AlphaBlend_0 _S356 = { _S353[_S354], Splat_2D_AlphaBlend_x24_syn_dzero_0() };

#line 87
                DiffPair_vectorx3Cfloatx2C2x3E_0 _S357 = { _S336, make_float2 (0.0f) };

#line 87
                DiffPair_float_0 _S358 = { _S355, 0.0f };

#line 87
                DiffPair_vectorx3Cfloatx2C4x3E_0 _S359 = s_fwd_evaluate_splat_0(_S356, _S357, _S358, H_5, W_5);


                if((_S359.primal_1.w) < 0.00392156885936856f)
                {

#line 91
                    j_2 = j_2 + int(1);

#line 91
                    local_n_contrib_6 = local_n_contrib_7;

#line 83
                    continue;
                }

#line 83
                DiffPair_vectorx3Cfloatx2C4x3E_0 _S360 = { curr_pixel_state_3, s_diff_curr_pixel_state_1 };

#line 83
                DiffPair_vectorx3Cfloatx2C4x3E_0 _S361 = { _S359.primal_1, _S359.differential_0 };

#line 93
                DiffPair_vectorx3Cfloatx2C4x3E_0 _S362 = s_fwd_update_pixel_state_0(_S360, _S361);


                if((_S362.primal_1.w) < 0.00009999999747379f)
                {
                    int _S363 = local_n_contrib_7 - int(1);

#line 98
                    thread_active_3 = false;

#line 98
                    local_n_contrib_5 = _S363;

                    break;
                }

#line 100
                curr_pixel_state_3 = _S362.primal_1;

#line 100
                s_diff_curr_pixel_state_1 = _S362.differential_0;

#line 83
                j_2 = j_2 + int(1);

#line 83
                local_n_contrib_6 = local_n_contrib_7;

#line 83
            }

#line 83
            thread_active_2 = thread_active_3;

#line 83
            local_n_contrib_4 = local_n_contrib_5;

#line 82
        }
        else
        {

#line 82
            curr_pixel_state_3 = curr_pixel_state_2;

#line 82
            s_diff_curr_pixel_state_1 = s_diff_curr_pixel_state_0;

#line 82
        }

#line 105
        int splats_left_to_process_5 = splats_left_to_process_4 - _S346;

#line 70
        int _S364 = i_11 + int(1);

#line 70
        curr_pixel_state_2 = curr_pixel_state_3;

#line 70
        s_diff_curr_pixel_state_0 = s_diff_curr_pixel_state_1;

#line 70
        i_11 = _S364;

#line 70
        splats_left_to_process_4 = splats_left_to_process_5;

#line 70
    }

#line 108
    if(is_inside_4)
    {

#line 109
        (n_contributors_5).store<int>((pix_coord_5.y), (_S339), (0U), (local_n_contrib_4));

#line 108
    }

#line 108
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S365 = { curr_pixel_state_2, s_diff_curr_pixel_state_0 };

    return _S365;
}


#line 110
__device__ void s_fwd_splat_tiled_0(TensorView sorted_gauss_idx_6, TensorView tile_ranges_3, DiffTensorView_0 xyz_vs_12, DiffTensorView_0 inv_cov_vs_12, DiffTensorView_0 opacity_12, DiffTensorView_0 rgb_12, DiffTensorView_0 distance_mu_12, DiffTensorView_0 distance_sigma_12, TensorView distance_7, DiffTensorView_0 output_img_3, TensorView n_contributors_6, int grid_height_3, int grid_width_3, int tile_height_6, int tile_width_6)
{

#line 262
    uint3  _S366 = ((blockIdx));

    uint2  pix_coord_6 = uint2 {(_S366 * ((blockDim)) + ((threadIdx))).x, (_S366 * ((blockDim)) + ((threadIdx))).y};

    uint tile_idx_1 = _S366.y * uint(grid_width_3) + _S366.x;
    int _S367 = ((tile_ranges_3).load<int>((tile_idx_1), (0U)));

#line 267
    uint tile_idx_start_5 = uint(_S367);
    int _S368 = ((tile_ranges_3).load<int>((tile_idx_1), (1U)));

#line 268
    uint tile_idx_end_5 = uint(_S368);

    uint _S369 = pix_coord_6.x;

#line 270
    uint _S370 = DiffTensorView_size_0(output_img_3, 1U);

#line 270
    bool is_inside_5;

#line 270
    if(_S369 < _S370)
    {

#line 270
        is_inside_5 = (pix_coord_6.y) < (DiffTensorView_size_0(output_img_3, 0U));

#line 270
    }
    else
    {

#line 270
        is_inside_5 = false;

#line 270
    }
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S371 = s_fwd_alpha_blend_0(sorted_gauss_idx_6, xyz_vs_12, inv_cov_vs_12, opacity_12, rgb_12, distance_mu_12, distance_sigma_12, distance_7, output_img_3, n_contributors_6, pix_coord_6, tile_idx_start_5, tile_idx_end_5, uint(tile_height_6), uint(tile_width_6), DiffTensorView_size_0(output_img_3, 0U), _S370);

#line 290
    if(is_inside_5)
    {

#line 291
        uint _S372 = pix_coord_6.y;

#line 291
        DiffPair_float_0 _S373 = { _S371.primal_1.x, _S371.differential_0.x };

#line 291
        DiffTensorView_storeOnce_forward_0(output_img_3, make_uint3 (_S372, _S369, 0U), _S373);

#line 291
        DiffPair_float_0 _S374 = { _S371.primal_1.y, _S371.differential_0.y };
        DiffTensorView_storeOnce_forward_0(output_img_3, make_uint3 (_S372, _S369, 1U), _S374);

#line 292
        DiffPair_float_0 _S375 = { _S371.primal_1.z, _S371.differential_0.z };
        DiffTensorView_storeOnce_forward_0(output_img_3, make_uint3 (_S372, _S369, 2U), _S375);

#line 293
        DiffPair_float_0 _S376 = { _S371.primal_1.w, _S371.differential_0.w };
        DiffTensorView_storeOnce_forward_0(output_img_3, make_uint3 (_S372, _S369, 3U), _S376);

#line 290
    }

#line 296
    return;
}


#line 296
extern "C" {
__global__ void __kernel__splat_tiled_fwd_diff(TensorView sorted_gauss_idx_7, TensorView tile_ranges_4, DiffTensorView_0 xyz_vs_13, DiffTensorView_0 inv_cov_vs_13, DiffTensorView_0 opacity_13, DiffTensorView_0 rgb_13, DiffTensorView_0 distance_mu_13, DiffTensorView_0 distance_sigma_13, TensorView distance_8, DiffTensorView_0 output_img_4, TensorView n_contributors_7, int grid_height_4, int grid_width_4, int tile_height_7, int tile_width_7)
{

#line 296
    s_fwd_splat_tiled_0(sorted_gauss_idx_7, tile_ranges_4, xyz_vs_13, inv_cov_vs_13, opacity_13, rgb_13, distance_mu_13, distance_sigma_13, distance_8, output_img_4, n_contributors_7, grid_height_4, grid_width_4, tile_height_7, tile_width_7);

#line 296
    return;
}

}

#line 246
__global__ void __kernel__splat_tiled(TensorView sorted_gauss_idx_8, TensorView tile_ranges_5, DiffTensorView_0 xyz_vs_14, DiffTensorView_0 inv_cov_vs_14, DiffTensorView_0 opacity_14, DiffTensorView_0 rgb_14, DiffTensorView_0 distance_mu_14, DiffTensorView_0 distance_sigma_14, TensorView distance_9, DiffTensorView_0 output_img_5, TensorView n_contributors_8, int grid_height_5, int grid_width_5, int tile_height_8, int tile_width_8)
{

#line 262
    uint3  _S377 = ((blockIdx));

    uint2  pix_coord_7 = uint2 {(_S377 * ((blockDim)) + ((threadIdx))).x, (_S377 * ((blockDim)) + ((threadIdx))).y};

    uint tile_idx_2 = _S377.y * uint(grid_width_5) + _S377.x;
    int _S378 = ((tile_ranges_5).load<int>((tile_idx_2), (0U)));

#line 267
    uint tile_idx_start_6 = uint(_S378);
    int _S379 = ((tile_ranges_5).load<int>((tile_idx_2), (1U)));

#line 268
    uint tile_idx_end_6 = uint(_S379);

    uint _S380 = pix_coord_7.x;

#line 270
    uint _S381 = DiffTensorView_size_0(output_img_5, 1U);

#line 270
    bool is_inside_6;

#line 270
    if(_S380 < _S381)
    {

#line 270
        is_inside_6 = (pix_coord_7.y) < (DiffTensorView_size_0(output_img_5, 0U));

#line 270
    }
    else
    {

#line 270
        is_inside_6 = false;

#line 270
    }
    float4  pixel_state_0 = alpha_blend_0(sorted_gauss_idx_8, xyz_vs_14, inv_cov_vs_14, opacity_14, rgb_14, distance_mu_14, distance_sigma_14, distance_9, output_img_5, n_contributors_8, pix_coord_7, tile_idx_start_6, tile_idx_end_6, uint(tile_height_8), uint(tile_width_8), DiffTensorView_size_0(output_img_5, 0U), _S381);

#line 290
    if(is_inside_6)
    {

#line 291
        uint _S382 = pix_coord_7.y;

#line 291
        DiffTensorView_storeOnce_0(output_img_5, make_uint3 (_S382, _S380, 0U), pixel_state_0.x);
        DiffTensorView_storeOnce_0(output_img_5, make_uint3 (_S382, _S380, 1U), pixel_state_0.y);
        DiffTensorView_storeOnce_0(output_img_5, make_uint3 (_S382, _S380, 2U), pixel_state_0.z);
        DiffTensorView_storeOnce_0(output_img_5, make_uint3 (_S382, _S380, 3U), pixel_state_0.w);

#line 290
    }

#line 296
    return;
}

