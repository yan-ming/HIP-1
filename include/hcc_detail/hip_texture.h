/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma once
/**
 *  @file  hip_kalmar_texture.h
 *  @brief HIP C++ Texture API for hcc compiler
 */

#include <limits.h> 

#include <hip_runtime.h>

//----
//Texture - TODO - likely need to move this to a separate file only included with kernel compilation.
#define hipTextureType1D 1

typedef enum {
  hipChannelFormatKindSigned = 0,
  hipChannelFormatKindUnsigned,
  hipChannelFormatKindFloat,
  hipChannelFormatKindNone

} hipChannelFormatKind;

typedef struct hipChannelFormatDesc {
  int x;
  int y;
  int z;
  int w;
  hipChannelFormatKind f;
} hipChannelFormatDesc;

typedef enum hipTextureReadMode 
{
  hipReadModeElementType,  ///< Read texture as specified element type
//! @warning cudaReadModeNormalizedFloat is not supported.
} hipTextureReadMode;

typedef enum hipTextureFilterMode 
{
    hipFilterModePoint,  ///< Point filter mode.
//! @warning cudaFilterModeLinear is not supported.
} hipTextureFilterMode;

struct textureReference {
    hipTextureFilterMode filterMode;
    bool                 normalized;
    hipChannelFormatDesc channelDesc;
};

template <class T, int texType=hipTextureType1D, enum hipTextureReadMode=hipReadModeElementType>
struct texture : public textureReference {

    const T * _dataPtr;  // pointer to underlying data.

    //texture() : filterMode(hipFilterModePoint), normalized(false), _dataPtr(NULL) {};
    unsigned int width;
    unsigned int height;

};

typedef struct hipArray {
  unsigned int width;
  unsigned int height;
  hipChannelFormatKind f;
  void* data; //FIXME: generalize this
} hipArray;





#define tex1Dfetch(_tex, _addr) (_tex._dataPtr[_addr])

#define tex2D(_tex, _dx, _dy) \
  _tex._dataPtr[(unsigned int)_dx + (unsigned int)_dy*(_tex.width)]

hipChannelFormatDesc hipCreateChannelDesc(int x=0, int y=0, int z=0, int w=0, hipChannelFormatKind f=hipChannelFormatKindFloat);

hipError_t hipMallocArray(hipArray** array, const hipChannelFormatDesc* desc,
                          size_t width, size_t height = 0, unsigned int flags = 0);

hipError_t hipFreeArray(hipArray* array);
  //
// dpitch, spitch, and width in bytes
hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind);

// wOffset, width, and spitch in bytes
hipError_t hipMemcpy2DToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
                              size_t spitch, size_t width, size_t height, hipMemcpyKind kind);

hipError_t hipMemcpyToArray(hipArray* dst, size_t wOffset, size_t hOffset,
                            const void* src, size_t count, hipMemcpyKind kind);


/**
 *  @addtogroup API HIP API
 *  @{
 *
 *  Defines the HIP API.  See the individual sections for more information.
 */

// These are C++ APIs - maybe belong in separate file.
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Texture Texture Reference Management
 *  @{
 *
 * 
 *  @warning The HIP texture API implements a small subset of full texture API.  Known limitations include:
 *  - Only point sampling is supported.
 *  - Only C++ APIs are provided.
 *  - Many APIs and modes are not implemented. 
 *
 *  The HIP texture support is intended to allow use of texture cache on hardware where this is beneficial.
 *
 *  The following CUDA APIs are not currently supported:
 *  - cudaBindTexture2D 
 *  - cudaBindTextureToArray
 *  - cudaBindTextureToMipmappedArray 
 *  - cudaGetChannelDesc 
 *  - cudaGetTextureReference
 *
 */

// C API:
#if 0
hipChannelFormatDesc  hipBindTexture(size_t *offset, struct textureReference *tex, const void *devPtr, const struct hipChannelFormatDesc *desc, size_t size=UINT_MAX) 
{
    tex->_dataPtr = devPtr;
}
#endif

/*
 * @brief hipChannelFormatDesc
 **/
// TODO
template <class T>
hipChannelFormatDesc  hipCreateChannelDesc() 
{
    hipChannelFormatDesc desc;
    return desc;
}

/*
 * @brief hipBindTexture
 **/
// TODO-doc
template <class T, int dim, enum hipTextureReadMode readMode>
hipError_t  hipBindTexture(size_t *offset, 
                                     struct texture<T, dim, readMode> &tex, 
                                     const void *devPtr, 
                                     const struct hipChannelFormatDesc *desc, 
                                     size_t size=UINT_MAX) 
{
    tex._dataPtr = static_cast<const T*>(devPtr);

    return hipSuccess;
}


/*
 * @brief hipBindTexture
 **/
// TODO-doc
template <class T, int dim, enum hipTextureReadMode readMode>
hipError_t  hipBindTexture(size_t *offset, 
                                     struct texture<T, dim, readMode> &tex, 
                                     const void *devPtr, 
                                     size_t size=UINT_MAX) 
{
    return  hipBindTexture(offset, tex, devPtr, &tex.channelDesc, size);
}

template <class T, int dim, enum hipTextureReadMode readMode>
hipError_t hipBindTextureToArray(struct texture<T, dim, readMode> &tex, hipArray* array) {
  tex.width = array->width;
  tex.height = array->height;
  tex._dataPtr = static_cast<const T*>(array->data);
  return hipSuccess;
}


/*
 * @brief hipUnbindTexture
 **/
// TODO-doc
template <class T, int dim, enum hipTextureReadMode readMode>
hipError_t  hipUnbindTexture(struct texture<T, dim, readMode> &tex)
{
    tex._dataPtr = NULL;

    return hipSuccess;
}



// doxygen end Texture
/**
 * @}  
 */


// End doxygen API:
/**
 *   @}
 */
