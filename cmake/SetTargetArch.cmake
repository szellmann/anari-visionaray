include(CheckCXXSourceCompiles)

function(check_cxx_source_compiles_with_flags SRC FLGS OUT_VAR)
  set(prev_flags "${CMAKE_REQUIRED_FLAGS}")
  set(CMAKE_REQUIRED_FLAGS "${FLGS}")
  check_cxx_source_compiles("${SRC}" ${OUT_VAR})
  set(CMAKE_REQUIRED_FLAGS "${prev_flags}")
endfunction()

function (set_target_arch TARGET)
  if (MSVC)
    set(SSE4_1_FLAGS "/arch:AVX2") # also enables SSE4.1
    set(AVX2_FLAGS "/arch:AVX2")
  else()
    set(SSE4_1_FLAGS "-msse4.1")
    set(AVX2_FLAGS "-mavx2")
  endif()

  check_cxx_source_compiles_with_flags("
    #if defined(__x86_64__) || defined(_M_X64)
    #include <smmintrin.h>
    int main() {
      __m128i a = _mm_set1_epi32(1);
      __m128i b = _mm_set1_epi32(1);
      __m128i c = _mm_mullo_epi32(a, b);
      return 0;
    }
    #endif
    " ${SSE4_1_FLAGS} SSE4_1_SUPPORTED)

  check_cxx_source_compiles_with_flags("
    #if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    int main() {
      __m256 v = _mm256_set1_ps(1.f);
      return 0;
    }
    #endif
    " ${AVX2_FLAGS} AVX2_SUPPORTED)

  # arm neon should be included automatically on 64-bit systems
  check_cxx_source_compiles("
    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    int main() {
      float32x4_t v = vdupq_n_f32(1.f);
      return 0;
    }
    #endif
    " ARM_NEON_SUPPORTED)

  if (AVX2_SUPPORTED)
    target_compile_options(${TARGET} PRIVATE ${AVX2_FLAGS})  
    target_compile_definitions(${TARGET} PRIVATE WITH_AVX2=1)
  elseif (SSE4_1_SUPPORTED)
    target_compile_options(${TARGET} PRIVATE ${SSE4_1_FLAGS})  
    target_compile_definitions(${TARGET} PRIVATE WITH_SSE4_1=1)
  elseif (ARM_NEON_SUPPORTED)
    target_compile_definitions(${TARGET} PRIVATE WITH_ARM_NEON=1)
  endif()
endfunction()
