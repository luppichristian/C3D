#pragma once
#include <stdbool.h>
#include <stddef.h>

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

//
// Errors (thread local)
//

typedef struct {
  const char* filename;
  const char* function;
  size_t line;
} C3DErrorLoc;

typedef enum {
  C3D_ERROR_NONE = 0,
  C3D_ERROR_INVALID_ARGUMENT,
  C3D_ERROR_UNSUPPORTED_FORMAT,
  C3D_ERROR_OUT_OF_MEMORY,
  C3D_ERROR_CUDA,
} C3DErrorID;

C3D_API void _c3dThrowError(C3DErrorID id, const char* desc, C3DErrorLoc loc);
#define c3dThrowError(id, desc) _c3dThrowError(id, desc, C3DErrorLoc {__FILE__, __FUNCTION__, __LINE__})

C3D_API C3DErrorID c3dGetErrorID(void);
C3D_API const char* c3dGetErrorDesc(void);

typedef void C3DErrorCallback(C3DErrorID id, const char* desc, C3DErrorLoc loc);
C3D_API void c3dSetErrorCallback(C3DErrorCallback* callback);

//
// Textures
//

typedef enum {
  C3D_TEXTURE_FORMAT_RGBA8,
  C3D_TEXTURE_FORMAT_BGRA8,
} C3DTextureFormat;

typedef struct {
  size_t width;
  size_t height;
  size_t depth;
  C3DTextureFormat format;
} C3DTextureInfo;

typedef struct C3DTexture C3DTexture;

C3D_API C3DTexture* c3dCreateTexture(const C3DTextureInfo* info);
C3D_API bool c3dDeleteTexture(C3DTexture* texture);
C3D_API bool c3dReadTexture(C3DTexture* texture, size_t offset, size_t size, void* buffer);
C3D_API bool c3dWriteTexture(C3DTexture* texture, size_t offset, size_t size, void* buffer);
C3D_API bool c3dGetTextureInfo(C3DTexture* texture, C3DTextureInfo* info);
