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
C3D_API bool c3dFillTexture(C3DTexture* texture, size_t offset, size_t size, void* texel);
C3D_API bool c3dClearTexture(C3DTexture* texture, void* texel);
C3D_API bool c3dGetTextureInfo(C3DTexture* texture, C3DTextureInfo* info);
C3D_API bool c3dResizeTexture(C3DTexture* texture, size_t width, size_t height, size_t depth);

//
// Index buffer
//

typedef enum {
  C3D_INDEX_SIZE_8,
  C3D_INDEX_SIZE_16,
  C3D_INDEX_SIZE_32,
} C3DIndexSize;

typedef struct {
  C3DIndexSize indexSize;
  size_t indexCap;
} C3DIndexBufferInfo;

typedef struct C3DIndexBuffer C3DIndexBuffer;

C3D_API C3DIndexBuffer* c3dCreateIndexBuffer(const C3DIndexBufferInfo* info);
C3D_API bool c3dDeleteIndexBuffer(C3DIndexBuffer* indexBuffer);
C3D_API bool c3dResizeIndexBuffer(C3DIndexBuffer* indexBuffer, size_t indexCap);
C3D_API bool c3dGetIndexBufferInfo(C3DIndexBuffer* indexBuffer, C3DIndexBufferInfo* info);
C3D_API bool c3dReadIndexBuffer(C3DIndexBuffer* indexBuffer, size_t offset, size_t size, void* buffer);
C3D_API bool c3dWriteIndexBuffer(C3DIndexBuffer* indexBuffer, size_t offset, size_t size, void* buffer);
C3D_API bool c3dFillIndexBuffer(C3DIndexBuffer* indexBuffer, size_t offset, size_t size, void* idx);
C3D_API bool c3dClearIndexBuffer(C3DIndexBuffer* indexBuffer, void* idx);

//
// Vertex buffer
//

typedef struct {
  float pos[4];
  float col[4];
  float uv[2];
  int texid;
} C3DVertex;

typedef struct {
  size_t vertexCap;
} C3DVertexBufferInfo;

typedef struct C3DVertexBuffer C3DVertexBuffer;

C3D_API C3DVertexBuffer* c3dCreateVertexBuffer(const C3DVertexBufferInfo* info);
C3D_API bool c3dDeleteVertexBuffer(C3DVertexBuffer* vertexBuffer);
C3D_API bool c3dResizeVertexBuffer(C3DVertexBuffer* vertexBuffer, size_t indexCap);
C3D_API bool c3dGetVertexBufferInfo(C3DVertexBuffer* vertexBuffer, C3DVertexBufferInfo* info);
C3D_API bool c3dReadVertexBuffer(C3DVertexBuffer* vertexBuffer, size_t offset, size_t size, void* buffer);
C3D_API bool c3dWriteVertexBuffer(C3DVertexBuffer* vertexBuffer, size_t offset, size_t size, void* buffer);
C3D_API bool c3dFillVertexBuffer(C3DVertexBuffer* vertexBuffer, size_t offset, size_t size, void* idx);
C3D_API bool c3dClearVertexBuffer(C3DVertexBuffer* vertexBuffer, void* idx);

//
// Command buffer
//

typedef enum {
  C3D_TOPOLOGY_LINE,
  C3D_TOPOLOGY_QUAD,
  C3D_TOPOLOGY_TRIANGLE,
} C3DTopology;

typedef enum {
  C3D_SAMPLER_POINT_CLAMP,
  C3D_SAMPLER_POINT_WRAP,
  C3D_SAMPLER_LINEAR_CLAMP,
  C3D_SAMPLER_LINEAR_WRAP,
} C3DSampler;

typedef enum {
  C3D_BLEND_MODE_NONE,
  C3D_BLEND_MODE_NORMAL,
  C3D_BLEND_MODE_ADDITIVE,
} C3DBlendMode;

typedef struct {
  C3DSampler sampler;
  C3DTexture* texture;
} C3DTextureBinding;

typedef struct {
  C3DTopology topology;
  C3DIndexBuffer* indexBuffer;
  size_t indexOffset;
  size_t indexBase;
  C3DVertexBuffer* vertexBuffer;
  size_t vertexOffset;
  size_t count;
} C3DDrawInfo;

typedef struct {
  C3DTexture* target;
  C3DBlendMode targetBlend;

  C3DTextureBinding* textureBindings;
  size_t textureBindCount;

} C3DRenderPassInfo;

typedef struct C3DCommandBuffer C3DCommandBuffer;

C3D_API C3DCommandBuffer* c3dCreateCommandBuffer(void);
C3D_API bool c3dDeleteCommandBuffer(C3DCommandBuffer* commandBuffer);
C3D_API bool c3dSubmitCommandBuffer(C3DCommandBuffer* commandBuffer);
C3D_API bool c3dCancelCommandBuffer(C3DCommandBuffer* commandBuffer);
C3D_API bool c3dBeginRenderPass(C3DCommandBuffer* commandBuffer, C3DRenderPassInfo* renderPass);
C3D_API bool c3dEndRenderPass(C3DCommandBuffer* commandBuffer);
C3D_API bool c3dDraw(C3DCommandBuffer* commandBuffer, const C3DDrawInfo* drawInfo);