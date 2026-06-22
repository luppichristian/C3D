#include <C3D.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tracy/TracyC.h>
#include "c3d_internal.h"

static bool c3dTryGrowDrawList(C3DCommandBuffer* commandBuffer, size_t minCap)
{
  TracyCZoneN(zone, "c3dTryGrowDrawList", 1);
  if (commandBuffer->drawCap >= minCap)
  {
    TracyCZoneEnd(zone);
    return true;
  }

  size_t newCap = commandBuffer->drawCap == 0 ? 4 : commandBuffer->drawCap * 2;
  while (newCap < minCap)
  {
    if (newCap > (SIZE_MAX / 2))
    {
      newCap = minCap;
      break;
    }

    newCap *= 2;
  }

  if (newCap > (SIZE_MAX / sizeof(C3DRecordedDraw)))
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw list size overflows size_t");
    TracyCZoneEnd(zone);
    return false;
  }

  C3DRecordedDraw* draws = (C3DRecordedDraw*)realloc(commandBuffer->draws, newCap * sizeof(C3DRecordedDraw));
  if (!draws)
  {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to grow command buffer draw list");
    TracyCZoneEnd(zone);
    return false;
  }

  commandBuffer->draws = draws;
  commandBuffer->drawCap = newCap;
  TracyCZoneEnd(zone);
  return true;
}

static void c3dResetRenderPassInfo(C3DRenderPassInfo* renderPass)
{
  free(renderPass->textureBindings);
  memset(renderPass, 0, sizeof(*renderPass));
}

void c3dResetCommandBuffer(C3DCommandBuffer* commandBuffer)
{
  commandBuffer->inRenderPass = false;
  commandBuffer->hasRenderPass = false;
  commandBuffer->drawCount = 0;
  c3dResetRenderPassInfo(&commandBuffer->renderPass);
}

static void c3dReleaseCommandBufferScratch(C3DCommandBuffer* commandBuffer)
{
  cudaFree(commandBuffer->depthBuffer);
  cudaFree(commandBuffer->linePrimitives);
  cudaFree(commandBuffer->trianglePrimitives);
  cudaFree(commandBuffer->textureViews);
  cudaFree(commandBuffer->tileCountsDevice);
  cudaFree(commandBuffer->tileOffsetsDevice);
  cudaFree(commandBuffer->tileIndicesDevice);
  free(commandBuffer->tileCountsHost);
  free(commandBuffer->tileOffsetsHost);
  commandBuffer->depthBuffer = nullptr;
  commandBuffer->depthCap = 0;
  commandBuffer->linePrimitives = nullptr;
  commandBuffer->linePrimitiveCap = 0;
  commandBuffer->trianglePrimitives = nullptr;
  commandBuffer->trianglePrimitiveCap = 0;
  commandBuffer->textureViews = nullptr;
  commandBuffer->textureViewCap = 0;
  commandBuffer->tileCountsDevice = nullptr;
  commandBuffer->tileOffsetsDevice = nullptr;
  commandBuffer->tileIndicesDevice = nullptr;
  commandBuffer->tileCountsHost = nullptr;
  commandBuffer->tileOffsetsHost = nullptr;
  commandBuffer->tileCountCap = 0;
  commandBuffer->tileOffsetCap = 0;
  commandBuffer->tileIndexCap = 0;
  commandBuffer->tileCountsHostCap = 0;
  commandBuffer->tileOffsetsHostCap = 0;
}

static bool c3dIsValidSampler(C3DSampler sampler)
{
  return sampler >= C3D_SAMPLER_POINT_CLAMP && sampler <= C3D_SAMPLER_LINEAR_WRAP;
}

static bool c3dIsValidTopology(C3DTopology topology)
{
  return topology >= C3D_TOPOLOGY_LINE && topology <= C3D_TOPOLOGY_TRIANGLE;
}

static bool c3dIsValidBlendMode(C3DBlendMode blendMode)
{
  return blendMode >= C3D_BLEND_MODE_NONE && blendMode <= C3D_BLEND_MODE_ADDITIVE;
}

static bool c3dIsColorTextureFormat(C3DTextureFormat format)
{
  return format == C3D_TEXTURE_FORMAT_RGBA8 || format == C3D_TEXTURE_FORMAT_BGRA8;
}

static bool c3dResolveViewport(C3DTexture* target, C3DViewport source, C3DViewport* resolved)
{
  if (!target || !resolved)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "target texture and viewport output must be non-null");
    return false;
  }

  *resolved = source;
  if (resolved->width == 0)
  {
    if (resolved->x > target->info.width)
    {
      c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "viewport x is out of bounds");
      return false;
    }

    resolved->width = target->info.width - resolved->x;
  }

  if (resolved->height == 0)
  {
    if (resolved->y > target->info.height)
    {
      c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "viewport y is out of bounds");
      return false;
    }

    resolved->height = target->info.height - resolved->y;
  }

  if (resolved->width == 0 || resolved->height == 0)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "viewport width and height must be greater than zero");
    return false;
  }

  if (resolved->x >= target->info.width || resolved->y >= target->info.height || resolved->width > target->info.width - resolved->x || resolved->height > target->info.height - resolved->y)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "viewport is out of bounds");
    return false;
  }

  return true;
}

static bool c3dValidateRenderPass(const C3DRenderPassInfo* renderPass)
{
  if (!renderPass)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "render pass info must be non-null");
    return false;
  }

  if (!renderPass->target)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "render pass target must be non-null");
    return false;
  }

  if (!c3dIsColorTextureFormat(renderPass->target->info.format))
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "render pass target format must be a color format");
    return false;
  }

  if (renderPass->target->info.depth != 1)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "render pass target depth must be 1");
    return false;
  }

  C3DViewport resolvedViewport = {0};
  if (!c3dResolveViewport(renderPass->target, renderPass->viewport, &resolvedViewport))
  {
    return false;
  }

  if (renderPass->depthTarget)
  {
    if (renderPass->depthTarget->info.format != C3D_TEXTURE_FORMAT_DEPTH64)
    {
      c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "render pass depth target must use C3D_TEXTURE_FORMAT_DEPTH64");
      return false;
    }

    if (renderPass->depthTarget->info.depth != 1 || renderPass->depthTarget->info.width != renderPass->target->info.width || renderPass->depthTarget->info.height != renderPass->target->info.height)
    {
      c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "render pass depth target dimensions must match the color target");
      return false;
    }
  }

  if (!c3dIsValidBlendMode(renderPass->targetBlend))
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "render pass blend mode is invalid");
    return false;
  }

  if (renderPass->textureBindCount != 0 && !renderPass->textureBindings)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture bindings must be non-null when textureBindCount is non-zero");
    return false;
  }

  for (size_t i = 0; i < renderPass->textureBindCount; ++i)
  {
    const C3DTextureBinding* binding = &renderPass->textureBindings[i];
    if (!binding->texture)
    {
      c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture binding texture must be non-null");
      return false;
    }

    if (!c3dIsColorTextureFormat(binding->texture->info.format))
    {
      c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture bindings must use color texture formats");
      return false;
    }

    if (!c3dIsValidSampler(binding->sampler))
    {
      c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture binding sampler is invalid");
      return false;
    }
  }

  return true;
}

static bool c3dCopyRenderPassInfo(C3DRenderPassInfo* destination, const C3DRenderPassInfo* source)
{
  *destination = *source;
  destination->textureBindings = nullptr;

  if (source->textureBindCount == 0)
  {
    return true;
  }

  if (source->textureBindCount > (SIZE_MAX / sizeof(C3DTextureBinding)))
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture binding list size overflows size_t");
    return false;
  }

  destination->textureBindings = (C3DTextureBinding*)malloc(source->textureBindCount * sizeof(C3DTextureBinding));
  if (!destination->textureBindings)
  {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate texture binding list");
    return false;
  }

  memcpy(destination->textureBindings, source->textureBindings, source->textureBindCount * sizeof(C3DTextureBinding));
  return true;
}

static bool c3dValidateDrawInfo(const C3DDrawInfo* drawInfo)
{
  if (!drawInfo)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw info must be non-null");
    return false;
  }

  if (!c3dIsValidTopology(drawInfo->topology))
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw topology is invalid");
    return false;
  }

  if (!drawInfo->indexBuffer || !drawInfo->vertexBuffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw buffers must be non-null");
    return false;
  }

  size_t indexStride = c3dGetIndexStride(drawInfo->indexSize);
  if (indexStride == 0)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw index size is invalid");
    return false;
  }

  if ((drawInfo->indexOffset % indexStride) != 0)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw index offset must be aligned to the index size");
    return false;
  }

  if ((drawInfo->vertexBuffer->info.size % sizeof(C3DVertex)) != 0)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw vertex buffer size must be aligned to C3DVertex");
    return false;
  }

  return true;
}

C3D_API C3DCommandBuffer* c3dCreateCommandBuffer(void)
{
  TracyCZoneN(zone, "c3dCreateCommandBuffer", 1);
  C3DCommandBuffer* commandBuffer = (C3DCommandBuffer*)malloc(sizeof(C3DCommandBuffer));
  if (!commandBuffer)
  {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate command buffer object");
    TracyCZoneEnd(zone);
    return nullptr;
  }

  memset(commandBuffer, 0, sizeof(C3DCommandBuffer));
  TracyCZoneEnd(zone);
  return commandBuffer;
}

C3D_API bool c3dDeleteCommandBuffer(C3DCommandBuffer* commandBuffer)
{
  TracyCZoneN(zone, "c3dDeleteCommandBuffer", 1);
  if (!commandBuffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer must be non-null");
    TracyCZoneEnd(zone);
    return false;
  }

  c3dResetRenderPassInfo(&commandBuffer->renderPass);
  c3dReleaseCommandBufferScratch(commandBuffer);
  free(commandBuffer->draws);
  free(commandBuffer);
  TracyCZoneEnd(zone);
  return true;
}

C3D_API bool c3dCancelCommandBuffer(C3DCommandBuffer* commandBuffer)
{
  TracyCZoneN(zone, "c3dCancelCommandBuffer", 1);
  if (!commandBuffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer must be non-null");
    TracyCZoneEnd(zone);
    return false;
  }

  c3dResetCommandBuffer(commandBuffer);
  TracyCZoneEnd(zone);
  return true;
}

C3D_API bool c3dBeginRenderPass(C3DCommandBuffer* commandBuffer, C3DRenderPassInfo* renderPass)
{
  TracyCZoneN(zone, "c3dBeginRenderPass", 1);
  if (!commandBuffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer must be non-null");
    TracyCZoneEnd(zone);
    return false;
  }

  if (commandBuffer->inRenderPass)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer already has an active render pass");
    TracyCZoneEnd(zone);
    return false;
  }

  if (!c3dValidateRenderPass(renderPass))
  {
    TracyCZoneEnd(zone);
    return false;
  }

  C3DRenderPassInfo resolvedRenderPass = *renderPass;
  if (!c3dResolveViewport(renderPass->target, renderPass->viewport, &resolvedRenderPass.viewport))
  {
    TracyCZoneEnd(zone);
    return false;
  }

  c3dResetCommandBuffer(commandBuffer);
  if (!c3dCopyRenderPassInfo(&commandBuffer->renderPass, &resolvedRenderPass))
  {
    TracyCZoneEnd(zone);
    return false;
  }

  commandBuffer->inRenderPass = true;
  commandBuffer->hasRenderPass = true;
  TracyCZoneEnd(zone);
  return true;
}

C3D_API bool c3dEndRenderPass(C3DCommandBuffer* commandBuffer)
{
  TracyCZoneN(zone, "c3dEndRenderPass", 1);
  if (!commandBuffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer must be non-null");
    TracyCZoneEnd(zone);
    return false;
  }

  if (!commandBuffer->inRenderPass)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer has no active render pass");
    TracyCZoneEnd(zone);
    return false;
  }

  commandBuffer->inRenderPass = false;
  TracyCZoneEnd(zone);
  return true;
}

C3D_API bool c3dDraw(C3DCommandBuffer* commandBuffer, const C3DDrawInfo* drawInfo)
{
  TracyCZoneN(zone, "c3dDraw", 1);
  if (!commandBuffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer must be non-null");
    TracyCZoneEnd(zone);
    return false;
  }

  if (!commandBuffer->inRenderPass)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw commands require an active render pass");
    TracyCZoneEnd(zone);
    return false;
  }

  if (!c3dValidateDrawInfo(drawInfo))
  {
    TracyCZoneEnd(zone);
    return false;
  }

  if (!c3dTryGrowDrawList(commandBuffer, commandBuffer->drawCount + 1))
  {
    TracyCZoneEnd(zone);
    return false;
  }

  commandBuffer->draws[commandBuffer->drawCount].info = *drawInfo;
  commandBuffer->drawCount += 1;
  TracyCZoneEnd(zone);
  return true;
}
