#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

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

// Extern C
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

#define C3D_ERROR_DESC_CAP 2048

typedef struct
{
  const char* filename;
  const char* function;
  size_t line;
} C3DErrorLoc;

typedef enum
{
  C3D_ERROR_NONE = 0,
  C3D_ERROR_INVALID_ARGUMENT,
  C3D_ERROR_UNSUPPORTED_FORMAT,
  C3D_ERROR_OUT_OF_MEMORY,
  C3D_ERROR_CUDA,
} C3DErrorID;

// Make sure to always call c3dThrowError
C3D_API void _c3dThrowError(C3DErrorID id, const char* desc, C3DErrorLoc loc);
#ifdef __cplusplus
#  define c3dThrowError(id, desc) _c3dThrowError(id, desc, C3DErrorLoc {__FILE__, __FUNCTION__, __LINE__})
#else
#  define c3dThrowError(id, desc) _c3dThrowError(id, desc, (C3DErrorLoc) {__FILE__, __FUNCTION__, __LINE__})
#endif

// Get last thrown error
C3D_API C3DErrorID c3dGetErrorID(void);
C3D_API const char* c3dGetErrorDesc(void);

// Error callback management
typedef void C3DErrorCallback(C3DErrorID id, const char* desc, C3DErrorLoc loc);
C3D_API void c3dSetErrorCallback(C3DErrorCallback* callback);
C3D_API C3DErrorCallback* c3dGetErrorCallback(void);

//
// Stage buffer
//

typedef struct
{
  const void* initBuffer;
  size_t initSize;
  size_t size;
} C3DStageBufferInfo;

typedef enum
{
  C3D_MEMORY_ACCESS_READ,
  C3D_MEMORY_ACCESS_WRITE,
  C3D_MEMORY_ACCESS_READ_WRITE,
} C3DMemoryAccess;

typedef struct C3DStageBuffer C3DStageBuffer;

C3D_API C3DStageBuffer* c3dCreateStageBuffer(const C3DStageBufferInfo* info);
C3D_API bool c3dDeleteStageBuffer(C3DStageBuffer* stageBuffer);
C3D_API bool c3dResizeStageBuffer(C3DStageBuffer* stageBuffer, size_t size);
C3D_API bool c3dGetStageBufferInfo(C3DStageBuffer* stageBuffer, C3DStageBufferInfo* info);
C3D_API bool c3dReadStageBuffer(C3DStageBuffer* stageBuffer, size_t offset, size_t size, void* buffer);
C3D_API bool c3dWriteStageBuffer(C3DStageBuffer* stageBuffer, size_t offset, size_t size, const void* buffer, bool cycle);
C3D_API void* c3dMapStageBuffer(C3DStageBuffer* stageBuffer, C3DMemoryAccess access, bool cycle);
C3D_API bool c3dUnmapStageBuffer(C3DStageBuffer* stageBuffer);

//
// Textures
//

typedef enum
{
  C3D_TEXTURE_FORMAT_RGBA8,
  C3D_TEXTURE_FORMAT_BGRA8,
  C3D_TEXTURE_FORMAT_DEPTH64,
} C3DTextureFormat;

typedef struct
{
  size_t width;
  size_t height;
  size_t depth;
  C3DTextureFormat format;
} C3DTextureInfo;

typedef struct C3DTexture C3DTexture;

C3D_API C3DTexture* c3dCreateTexture(const C3DTextureInfo* info);
C3D_API bool c3dDeleteTexture(C3DTexture* texture);
C3D_API bool c3dReadTexture(C3DTexture* texture, size_t textureOffset, size_t size, C3DStageBuffer* stageBuffer, size_t stageOffset);
C3D_API bool c3dWriteTexture(C3DTexture* texture, size_t textureOffset, size_t size, C3DStageBuffer* stageBuffer, size_t stageOffset, bool cycle);
C3D_API bool c3dFillTexture(C3DTexture* texture, size_t offset, size_t size, void* texel, bool cycle);
C3D_API bool c3dClearTexture(C3DTexture* texture, void* texel, bool cycle);
C3D_API bool c3dGetTextureInfo(C3DTexture* texture, C3DTextureInfo* info);
C3D_API bool c3dResizeTexture(C3DTexture* texture, size_t width, size_t height, size_t depth);

//
// Swapchain
//

typedef enum
{
  C3D_PRESENT_MODE_FIFO,
  C3D_PRESENT_MODE_MAILBOX,
  C3D_PRESENT_MODE_IMMEDIATE,
} C3DPresentMode;

typedef struct
{
  const void* pixels;
  size_t width;
  size_t height;
  size_t rowPitch;
  C3DTextureFormat format;
  uint64_t frameId;
} C3DPresentFrame;

typedef struct
{
  void* userData;
  bool (*present)(void* userData, const C3DPresentFrame* frame);
} C3DPresenterOps;

typedef struct
{
  size_t width;
  size_t height;
  C3DTextureFormat format;
  size_t imageCount;
  C3DPresentMode presentMode;
  C3DPresenterOps presenter;
} C3DSwapchainInfo;

typedef struct C3DSwapchain C3DSwapchain;

C3D_API C3DSwapchain* c3dCreateSwapchain(const C3DSwapchainInfo* info);
C3D_API bool c3dDeleteSwapchain(C3DSwapchain* swapchain);
C3D_API bool c3dResizeSwapchain(C3DSwapchain* swapchain, size_t width, size_t height);
C3D_API bool c3dGetSwapchainInfo(C3DSwapchain* swapchain, C3DSwapchainInfo* info);
C3D_API bool c3dAcquireNextTexture(C3DSwapchain* swapchain, C3DTexture** texture);
C3D_API bool c3dPresentSwapchain(C3DSwapchain* swapchain);

//
// Buffers
//

typedef enum
{
  C3D_INDEX_SIZE_8,
  C3D_INDEX_SIZE_16,
  C3D_INDEX_SIZE_32,
} C3DIndexSize;

typedef struct
{
  size_t size;
} C3DBufferInfo;

typedef struct C3DBuffer C3DBuffer;

C3D_API C3DBuffer* c3dCreateBuffer(const C3DBufferInfo* info);
C3D_API bool c3dDeleteBuffer(C3DBuffer* buffer);
C3D_API bool c3dResizeBuffer(C3DBuffer* buffer, size_t size);
C3D_API bool c3dGetBufferInfo(C3DBuffer* buffer, C3DBufferInfo* info);
C3D_API bool c3dReadBuffer(C3DBuffer* buffer, size_t bufferOffset, size_t size, C3DStageBuffer* stageBuffer, size_t stageOffset);
C3D_API bool c3dWriteBuffer(C3DBuffer* buffer, size_t bufferOffset, size_t size, C3DStageBuffer* stageBuffer, size_t stageOffset, bool cycle);
C3D_API bool c3dBufferCopy(C3DBuffer* destination, size_t destinationOffset, C3DBuffer* source, size_t sourceOffset, size_t size, bool cycle);

//
// Vertex layout
//

typedef struct
{
  float pos[4];
  float col[4];
  float uv[2];
  int texid;
} C3DVertex;

//
// Command buffer
//

typedef enum
{
  C3D_TOPOLOGY_LINE,
  C3D_TOPOLOGY_QUAD,
  C3D_TOPOLOGY_TRIANGLE,
} C3DTopology;

typedef enum
{
  C3D_SAMPLER_POINT_CLAMP,
  C3D_SAMPLER_POINT_WRAP,
  C3D_SAMPLER_LINEAR_CLAMP,
  C3D_SAMPLER_LINEAR_WRAP,
} C3DSampler;

typedef enum
{
  C3D_BLEND_MODE_NONE,
  C3D_BLEND_MODE_NORMAL,
  C3D_BLEND_MODE_ADDITIVE,
} C3DBlendMode;

typedef enum
{
  C3D_LOAD_OP_LOAD,
  C3D_LOAD_OP_CLEAR,
} C3DLoadOp;

typedef struct
{
  C3DSampler sampler;
  C3DTexture* texture;
} C3DTextureBinding;

typedef struct
{
  size_t x;
  size_t y;
  size_t width;
  size_t height;
} C3DViewport;

typedef struct
{
  C3DTopology topology;
  C3DBuffer* indexBuffer;
  C3DIndexSize indexSize;
  size_t indexOffset;
  size_t indexBase;
  C3DBuffer* vertexBuffer;
  size_t vertexOffset;
  size_t count;
} C3DDrawInfo;

typedef struct
{
  C3DTexture* target;
  bool cycleTarget;
  C3DLoadOp targetLoadOp;
  uint8_t targetClearColor[4];
  C3DTexture* depthTarget;
  bool cycleDepthTarget;
  C3DLoadOp depthLoadOp;
  C3DViewport viewport;
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
