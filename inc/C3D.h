#pragma once

// Define C3D_IMPORT and C3D_EXPORT
#if defined(__INTELLISENSE__)
#  define C3D_IMPORT
#  define C3D_EXPORT
#elif defined(_WIN32) || defined(__CYGWIN__)
#  if defined(__GNUC__) || defined(__clang__)
#    define C3D_IMPORT __attribute__((dllimport))
#    define C3D_EXPORT __attribute__((dllexport))
#  else
#    define C3D_IMPORT __declspec(dllimport)
#    define C3D_EXPORT __declspec(dllexport)
#  endif
#else
#  define C3D_IMPORT
#  define C3D_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#  define C3D_EXTERN_C extern "C"
#else
#  define C3D_EXTERN_C extern
#endif

// Define C3D_API
#if defined(C3D_EXPORT_LIB)
#  define C3D_API C3D_EXTERN_C C3D_EXPORT
#else
#  define C3D_API C3D_EXTERN_C C3D_IMPORT
#endif

// Example function
C3D_API void C3D_HelloFromCUDA(void);
