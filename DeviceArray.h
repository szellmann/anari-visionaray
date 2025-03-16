
#pragma once

// std
#include <mutex>
#include <vector>
// ours
#include "DeviceCopyableObjects.h"

#ifdef WITH_CUDA
// cuda
#include <cuda_runtime.h>
// visionaray
#include "visionaray/cuda/safe_call.h"
#elif defined(WITH_HIP)
// cuda
#include <hip/hip_runtime.h>
// visionaray
#include "visionaray/hip/safe_call.h"
#endif

namespace visionaray {

#ifdef WITH_CUDA

// ==================================================================
// dynamic array for cuda device data
// ==================================================================

template <typename T>
struct DeviceArray
{
 public:
  typedef T value_type;

  DeviceArray() = default;

  ~DeviceArray()
  {
    CUDA_SAFE_CALL(cudaFree(devicePtr));
    devicePtr = nullptr;
    len = 0;
  }

  DeviceArray(size_t n)
  {
    CUDA_SAFE_CALL(cudaMalloc(&devicePtr, n*sizeof(T)));
    len = n;
  }

  DeviceArray(const DeviceArray &rhs)
  {
    if (&rhs != this) {
      CUDA_SAFE_CALL(cudaMalloc(&devicePtr, rhs.len*sizeof(T)));
      CUDA_SAFE_CALL(cudaMemcpy(devicePtr, rhs.devicePtr, len*sizeof(T),
                                cudaMemcpyDeviceToDevice));
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      len = rhs.len;
    }
  }

  DeviceArray(DeviceArray &&rhs)
  {
    if (&rhs != this) {
      CUDA_SAFE_CALL(cudaMalloc(&devicePtr, rhs.len*sizeof(T)));
      CUDA_SAFE_CALL(cudaMemcpy(devicePtr, rhs.devicePtr, len*sizeof(T),
                                cudaMemcpyDeviceToDevice));
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      CUDA_SAFE_CALL(cudaFree(rhs.devicePtr));
      len = rhs.len;
      rhs.devicePtr = nullptr;
      rhs.len = 0;
    }
  }

  DeviceArray &operator=(const DeviceArray &rhs)
  {
    if (&rhs != this) {
      CUDA_SAFE_CALL(cudaMalloc(&devicePtr, rhs.len*sizeof(T)));
      CUDA_SAFE_CALL(cudaMemcpy(devicePtr, rhs.devicePtr, len*sizeof(T),
                                cudaMemcpyDeviceToDevice));
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      len = rhs.len;
    }
    return *this;
  }

  DeviceArray &operator=(DeviceArray &&rhs)
  {
    if (&rhs != this) {
      CUDA_SAFE_CALL(cudaMalloc(&devicePtr, rhs.len*sizeof(T)));
      CUDA_SAFE_CALL(cudaMemcpy(devicePtr, rhs.devicePtr, len*sizeof(T),
                                cudaMemcpyDeviceToDevice));
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      CUDA_SAFE_CALL(cudaFree(rhs.devicePtr));
      rhs.devicePtr = nullptr;
      rhs.len = 0;
    }
    return *this;
  }

  T *data()
  { return devicePtr; }

  const T *data() const
  { return devicePtr; }

  size_t size() const
  { return len; }

  void resize(size_t n)
  {
    if (n == len)
      return;

    T *temp{nullptr};
    if (devicePtr && len > 0) {
      CUDA_SAFE_CALL(cudaMalloc(&temp, len*sizeof(T)));
      CUDA_SAFE_CALL(cudaMemcpy(temp, devicePtr, len*sizeof(T),
                                cudaMemcpyDeviceToDevice));
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      CUDA_SAFE_CALL(cudaFree(devicePtr));
    }

    CUDA_SAFE_CALL(cudaMalloc(&devicePtr, n*sizeof(T)));

    if (temp) {
      CUDA_SAFE_CALL(cudaMemcpy(devicePtr, temp, std::min(n, len)*sizeof(T),
                                cudaMemcpyDeviceToDevice));
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      CUDA_SAFE_CALL(cudaFree(temp));
    }

    len = n;
  }

 private:
  T *devicePtr{nullptr};
  size_t len{0};
};

#elif defined(WITH_HIP)

// ==================================================================
// dynamic array for hip device data
// ==================================================================

template <typename T>
struct DeviceArray
{
 public:
  typedef T value_type;

  DeviceArray() = default;

  ~DeviceArray()
  {
    HIP_SAFE_CALL(hipFree(devicePtr));
    devicePtr = nullptr;
    len = 0;
  }

  DeviceArray(size_t n)
  {
    HIP_SAFE_CALL(hipMalloc(&devicePtr, n*sizeof(T)));
    len = n;
  }

  DeviceArray(const DeviceArray &rhs)
  {
    if (&rhs != this) {
      HIP_SAFE_CALL(hipMalloc(&devicePtr, rhs.len*sizeof(T)));
      HIP_SAFE_CALL(hipMemcpy(devicePtr, rhs.devicePtr, len*sizeof(T),
                              hipMemcpyDeviceToDevice));
      HIP_SAFE_CALL(hipDeviceSynchronize());
      len = rhs.len;
    }
  }

  DeviceArray(DeviceArray &&rhs)
  {
    if (&rhs != this) {
      HIP_SAFE_CALL(hipMalloc(&devicePtr, rhs.len*sizeof(T)));
      HIP_SAFE_CALL(hipMemcpy(devicePtr, rhs.devicePtr, len*sizeof(T),
                              hipMemcpyDeviceToDevice));
      HIP_SAFE_CALL(hipDeviceSynchronize());
      HIP_SAFE_CALL(hipFree(rhs.devicePtr));
      len = rhs.len;
      rhs.devicePtr = nullptr;
      rhs.len = 0;
    }
  }

  DeviceArray &operator=(const DeviceArray &rhs)
  {
    if (&rhs != this) {
      HIP_SAFE_CALL(hipMalloc(&devicePtr, rhs.len*sizeof(T)));
      HIP_SAFE_CALL(hipMemcpy(devicePtr, rhs.devicePtr, len*sizeof(T),
                              hipMemcpyDeviceToDevice));
      HIP_SAFE_CALL(hipDeviceSynchronize());
      len = rhs.len;
    }
    return *this;
  }

  DeviceArray &operator=(DeviceArray &&rhs)
  {
    if (&rhs != this) {
      HIP_SAFE_CALL(hipMalloc(&devicePtr, rhs.len*sizeof(T)));
      HIP_SAFE_CALL(hipMemcpy(devicePtr, rhs.devicePtr, len*sizeof(T),
                              hipMemcpyDeviceToDevice));
      HIP_SAFE_CALL(hipDeviceSynchronize());
      HIP_SAFE_CALL(hipFree(rhs.devicePtr));
      rhs.devicePtr = nullptr;
      rhs.len = 0;
    }
    return *this;
  }

  T *data()
  { return devicePtr; }

  const T *data() const
  { return devicePtr; }

  size_t size() const
  { return len; }

  void resize(size_t n)
  {
    if (n == len)
      return;

    T *temp{nullptr};
    if (devicePtr && len > 0) {
      HIP_SAFE_CALL(hipMalloc(&temp, len*sizeof(T)));
      HIP_SAFE_CALL(hipMemcpy(temp, devicePtr, len*sizeof(T),
                              hipMemcpyDeviceToDevice));
      HIP_SAFE_CALL(hipDeviceSynchronize());
      HIP_SAFE_CALL(hipFree(devicePtr));
    }

    HIP_SAFE_CALL(hipMalloc(&devicePtr, n*sizeof(T)));

    if (temp) {
      HIP_SAFE_CALL(hipMemcpy(devicePtr, temp, std::min(n, len)*sizeof(T),
                              hipMemcpyDeviceToDevice));
      HIP_SAFE_CALL(hipDeviceSynchronize());
      HIP_SAFE_CALL(hipFree(temp));
    }

    len = n;
  }

 private:
  T *devicePtr{nullptr};
  size_t len{0};
};

#else

// ==================================================================
// dynamic array for device data, emulated on the host
// ==================================================================

template <typename T>
struct DeviceArray
{
 public:
  typedef T value_type;

  DeviceArray() = default;

  ~DeviceArray()
  {
    std::free(devicePtr);
    devicePtr = nullptr;
    len = 0;
  }

  DeviceArray(size_t n)
  {
    devicePtr = (T *)std::malloc(n*sizeof(T));
    len = n;
  }

  DeviceArray(const DeviceArray &rhs)
  {
    if (&rhs != this) {
      devicePtr = (T *)std::malloc(rhs.len*sizeof(T));
      std::memcpy(devicePtr, rhs.devicePtr, len*sizeof(T));
      len = rhs.len;
    }
  }

  DeviceArray(DeviceArray &&rhs)
  {
    if (&rhs != this) {
      devicePtr = (T *)std::malloc(rhs.len*sizeof(T));
      std::memcpy(devicePtr, rhs.devicePtr, len*sizeof(T));
      std::free(rhs.devicePtr);
      len = rhs.len;
      rhs.devicePtr = nullptr;
      rhs.len = 0;
    }
  }

  DeviceArray &operator=(const DeviceArray &rhs)
  {
    if (&rhs != this) {
      devicePtr = (T *)std::malloc(rhs.len*sizeof(T));
      std::memcpy(devicePtr, rhs.devicePtr, len*sizeof(T));
      len = rhs.len;
    }
    return *this;
  }

  DeviceArray &operator=(DeviceArray &&rhs)
  {
    if (&rhs != this) {
      devicePtr = (T *)std::malloc(rhs.len*sizeof(T));
      std::memcpy(devicePtr, rhs.devicePtr, len*sizeof(T));
      std::free(rhs.devicePtr);
      rhs.devicePtr = nullptr;
      rhs.len = 0;
    }
    return *this;
  }

  T *data()
  { return devicePtr; }

  const T *data() const
  { return devicePtr; }

  size_t size() const
  { return len; }

  void resize(size_t n)
  {
    if (n == len)
      return;

    T *temp{nullptr};
    if (devicePtr && len > 0) {
      temp = (T *)std::malloc(len*sizeof(T));
      std::memcpy(temp, devicePtr, len*sizeof(T));
      std::free(devicePtr);
    }

    devicePtr = (T *)std::malloc(n*sizeof(T));

    if (temp) {
      std::memcpy(devicePtr, temp, std::min(n, len)*sizeof(T));
      std::free(temp);
    }

    len = n;
  }

 private:
  T *devicePtr{nullptr};
  size_t len{0};
};

#endif

// ==================================================================
// host/device array
// ==================================================================

template <typename T>
struct HostDeviceArray : public std::vector<T>
{
 public:
  // TODO: assert trivially copyable
  typedef T value_type;
  typedef std::vector<T> Base;

  HostDeviceArray() = default;
  ~HostDeviceArray() = default;

  void *mapDevice()
  {
    deviceMapped = true;
    return devicePtr();
  }

  void unmapDevice()
  {
    updateOnHost();
    deviceMapped = false;
  }

  void set(size_t index, const T &value)
  {
    if (index >= Base::size())
      resize(index+1);

    std::unique_lock<std::mutex> l(mtx);
    updated = true;
    Base::operator[](index) = value;
  }

  void resize(size_t n)
  {
    std::unique_lock<std::mutex> l(mtx);
    Base::resize(n);
    updated = true;
  }

  void push_back(const T &value)
  {
    std::unique_lock<std::mutex> l(mtx);
    Base::push_back(value);
    updated = true;
  }

  void push_back(T &&value)
  {
    std::unique_lock<std::mutex> l(mtx);
    Base::push_back(value);
    updated = true;
  }

  void resize(size_t n, const T &value)
  {
    std::unique_lock<std::mutex> l(mtx);
    Base::resize(n, value);
    updated = true;
  }

  void reset(const void *data)
  {
    std::unique_lock<std::mutex> l(mtx);
    memcpy(Base::data(), data, Base::size() * sizeof(T));
    updated = true;
  }

  T &operator[](size_t i)
  {
    std::unique_lock<std::mutex> l(mtx);
    updated = true;
    return Base::operator[](i);
  }

  const T *hostPtr() const
  {
    return Base::data();
  }

  T *devicePtr()
  {
    updateOnDevice();
    return deviceData.data();
  }

 protected:
#if defined(WITH_CUDA) || defined(WITH_HIP)
  DeviceArray<T> deviceData;
#else
  Base deviceData;
#endif
  bool updated = true;
  bool deviceMapped = false;

 private:
  void updateOnDevice()
  {
    if (!updated)
      return;

    std::unique_lock<std::mutex> l(mtx);
    deviceData.resize(Base::size());
#ifdef WITH_CUDA
    CUDA_SAFE_CALL(cudaMemcpy(deviceData.data(),
                              Base::data(),
                              Base::size() * sizeof(T),
                              cudaMemcpyHostToDevice));
#elif defined(WITH_HIP)
    HIP_SAFE_CALL(hipMemcpy(deviceData.data(),
                            Base::data(),
                            Base::size() * sizeof(T),
                            hipMemcpyHostToDevice));
#else
    memcpy(deviceData.data(), Base::data(), Base::size() * sizeof(T));
#endif
    updated = false;
  }

  void updateOnHost()
  {
    std::unique_lock<std::mutex> l(mtx);
    Base::resize(deviceData.size());
#ifdef WITH_CUDA
    CUDA_SAFE_CALL(cudaMemcpy(Base::data(),
                              deviceData.data(),
                              Base::size() * sizeof(T),
                              cudaMemcpyDeviceToHost));
#elif defined(WITH_HIP)
    HIP_SAFE_CALL(hipMemcpy(Base::data(),
                            deviceData.data(),
                            Base::size() * sizeof(T),
                            hipMemcpyDeviceToHost));
#else
    memcpy(Base::data(), deviceData.data(), Base::size() * sizeof(T));
#endif
    updated = false; // !
  }

  std::mutex mtx;
};

// ==================================================================
// host/device array storing device object handles
// ==================================================================

typedef HostDeviceArray<DeviceObjectHandle> DeviceHandleArray;

// ==================================================================
// Array type capable of managing device-copyable objects via handles
// TODO: can this use HostDeviceArray internally?!
// ==================================================================

template <typename T>
struct DeviceObjectArray : private std::vector<T>
{
 public:
  typedef typename std::vector<T>::value_type value_type;
  typedef std::vector<T> Base;

  DeviceObjectArray() = default;
  ~DeviceObjectArray() = default;

  DeviceObjectHandle alloc(const T &obj)
  {
    std::unique_lock<std::mutex> l(mtx);
    Base::push_back(obj);
    updated = true;
    return (DeviceObjectHandle)(Base::size()-1);
  }

  void free(DeviceObjectHandle handle)
  {
    updated = true;
  }

  void update(DeviceObjectHandle handle, const T &obj)
  {
    std::unique_lock<std::mutex> l(mtx);
    Base::data()[handle] = obj;
    updated = true;
  }

  size_t size() const
  {
    return Base::size();
  }

  bool empty() const
  {
    return Base::empty();
  }

  void clear()
  {
    std::unique_lock<std::mutex> l(mtx);
    Base::clear();
    freeHandles.clear();
    updated = true;
  }

  const T &operator[](size_t index) const
  {
    return Base::operator[](index);
  }

  const T *hostPtr() const
  {
    return Base::data();
  }

  T *devicePtr()
  {
    if (updated) {
      std::unique_lock<std::mutex> l(mtx);
      deviceData.resize(Base::size());
#ifdef WITH_CUDA
      CUDA_SAFE_CALL(cudaMemcpy(deviceData.data(),
                     Base::data(),
                     Base::size() * sizeof(T),
                     cudaMemcpyHostToDevice));
#elif defined(WITH_HIP)
      HIP_SAFE_CALL(hipMemcpy(deviceData.data(),
                    Base::data(),
                    Base::size() * sizeof(T),
                    hipMemcpyHostToDevice));
#else
      // TODO: assert trivially copyable
      memcpy(deviceData.data(), Base::data(), Base::size() * sizeof(T));
#endif
      updated = false;
    }
    return deviceData.data();
  }

  std::vector<DeviceObjectHandle> freeHandles;
  DeviceArray<T> deviceData;
  bool updated = true;

  std::mutex mtx;
};

} // namespace visionaray
