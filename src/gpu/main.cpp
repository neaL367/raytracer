// Headless Vulkan compute: `rt_gpu fill` runs the gradient scaffold kernel,
// `rt_gpu normal` traces primary rays over the shared scene (scene/scene.h)
// with normal shading for CPU parity. No surface, no swapchain: instance ->
// GPU -> compute queue -> dispatch -> host readback -> existing PPM writer.
#include "../app/config.h"
#include "../core/bvh.h"
#include "../core/camera.h"
#include "../core/material.h"
#include "../core/quad.h"
#include "../core/sphere.h"
#include "../core/triangle.h"
#include "../io/ppm.h"
#include "../scene/scene.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#define VK_CHECK(x)                                                                            \
    do                                                                                         \
    {                                                                                          \
        VkResult vk_check_r = (x);                                                             \
        if (vk_check_r != VK_SUCCESS)                                                          \
        {                                                                                      \
            std::fprintf(stderr, "Vulkan error %d at %s:%d\n", vk_check_r, __FILE__, __LINE__); \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

namespace
{

struct Gpu
{
    VkInstance instance = nullptr;
    VkPhysicalDevice physical = nullptr;
    VkDevice device = nullptr;
    VkQueue queue = nullptr;
    uint32_t qfamily = 0;
    // Nanoseconds per timestamp tick; 0 when the queue cannot timestamp.
    float timestamp_period = 0.0f;
};

// Directory holding this executable: shaders ship next to the binary (see
// CMake POST_BUILD copy), so the app runs from anywhere.
std::filesystem::path exe_dir(const char *argv0)
{
#ifdef _WIN32
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n > 0)
        return std::filesystem::path(std::string(buf, n)).parent_path();
#endif
    return std::filesystem::path(argv0).parent_path();
}

std::vector<char> read_file(const std::filesystem::path &path)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
    {
        std::fprintf(stderr, "cannot open %s\n", path.string().c_str());
        std::exit(1);
    }
    std::vector<char> bytes(static_cast<size_t>(in.tellg()));
    in.seekg(0);
    in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return bytes;
}

// Queue family serving compute. Dedicated compute preferred over a shared
// graphics+compute one (async-friendly later) — but only if it can
// timestamp; otherwise the shared family wins so dispatches stay measurable.
uint32_t find_compute_family(VkPhysicalDevice device)
{
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, props.data());
    auto serves_compute = [&](uint32_t i) { return (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0; };
    auto shares_graphics = [&](uint32_t i) { return (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0; };
    auto can_stamp = [&](uint32_t i) { return props[i].timestampValidBits > 0; };
    for (uint32_t i = 0; i < count; ++i)
        if (serves_compute(i) && !shares_graphics(i) && can_stamp(i))
            return i;
    for (uint32_t i = 0; i < count; ++i)
        if (serves_compute(i) && can_stamp(i))
            return i;
    for (uint32_t i = 0; i < count; ++i)
        if (serves_compute(i))
            return i;
    std::fprintf(stderr, "no compute queue family\n");
    std::exit(1);
}

uint32_t find_memory_type(VkPhysicalDevice device, uint32_t bits, VkMemoryPropertyFlags flags)
{
    VkPhysicalDeviceMemoryProperties mem = {};
    vkGetPhysicalDeviceMemoryProperties(device, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    std::fprintf(stderr, "no suitable memory type\n");
    std::exit(1);
}

Gpu init_gpu()
{
    Gpu g;
    VkApplicationInfo app = {};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &g.instance));

    uint32_t gpu_count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(g.instance, &gpu_count, nullptr));
    if (gpu_count == 0)
    {
        std::fprintf(stderr, "no Vulkan physical devices\n");
        std::exit(1);
    }
    std::vector<VkPhysicalDevice> gpus(gpu_count);
    VK_CHECK(vkEnumeratePhysicalDevices(g.instance, &gpu_count, gpus.data()));
    g.physical = gpus[0];
    for (auto candidate : gpus)
    {
        VkPhysicalDeviceProperties props = {};
        vkGetPhysicalDeviceProperties(candidate, &props);
        std::printf("GPU: %s\n", props.deviceName);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            g.physical = candidate;
            break;
        }
    }
    VkPhysicalDeviceProperties chosen = {};
    vkGetPhysicalDeviceProperties(g.physical, &chosen);
    std::printf("using: %s\n", chosen.deviceName);

    g.qfamily = find_compute_family(g.physical);
    {
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(g.physical, &count, nullptr);
        std::vector<VkQueueFamilyProperties> props(count);
        vkGetPhysicalDeviceQueueFamilyProperties(g.physical, &count, props.data());
        if (props[g.qfamily].timestampValidBits > 0)
        {
            VkPhysicalDeviceProperties dev_props = {};
            vkGetPhysicalDeviceProperties(g.physical, &dev_props);
            g.timestamp_period = dev_props.limits.timestampPeriod;
        }
        std::printf("queue family %u (timestamps %s)\n", g.qfamily,
                    g.timestamp_period > 0.0f ? "on" : "off");
    }
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = g.qfamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci = {};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VK_CHECK(vkCreateDevice(g.physical, &dci, nullptr, &g.device));
    vkGetDeviceQueue(g.device, g.qfamily, 0, &g.queue);
    return g;
}

void shutdown_gpu(Gpu &g)
{
    vkDestroyDevice(g.device, nullptr);
    vkDestroyInstance(g.instance, nullptr);
}

struct Buffer
{
    VkBuffer handle = nullptr;
    VkDeviceMemory memory = nullptr;
    VkDeviceSize size = 0;
};

VkBufferUsageFlags storage_usage()
{
    return VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
}

// Explicit memory placement: host-visible for direct mapping, device-local
// for shader traffic (with TRANSFER bits when staging passes through).
Buffer make_buffer(const Gpu &g, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_props)
{
    Buffer b;
    b.size = size;
    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(g.device, &bci, nullptr, &b.handle));
    VkMemoryRequirements reqs = {};
    vkGetBufferMemoryRequirements(g.device, b.handle, &reqs);
    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = reqs.size;
    mai.memoryTypeIndex = find_memory_type(g.physical, reqs.memoryTypeBits, mem_props);
    VK_CHECK(vkAllocateMemory(g.device, &mai, nullptr, &b.memory));
    VK_CHECK(vkBindBufferMemory(g.device, b.handle, b.memory, 0));
    return b;
}

// Host-visible shortcut for debug/simple paths (fill, normal): mapped
// directly, no staging. Device reads/writes cross PCIe per access — fine
// for tests, wrong for the timed path.
Buffer make_mapped_buffer(const Gpu &g, VkDeviceSize size)
{
    return make_buffer(g, size, storage_usage(),
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

// Device-local buffer fed by a host-visible staging copy. One fence
// round-trip per call; upload happens once per run so batching buys nothing.
void upload_to_device(const Gpu &g, const Buffer &dst, const void *data, size_t bytes)
{
    Buffer staging = make_buffer(g, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void *mapped = nullptr;
    VK_CHECK(vkMapMemory(g.device, staging.memory, 0, bytes, 0, &mapped));
    std::memcpy(mapped, data, bytes);
    vkUnmapMemory(g.device, staging.memory);

    VkCommandPoolCreateInfo cpci = {};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = g.qfamily;
    VkCommandPool pool = nullptr;
    VK_CHECK(vkCreateCommandPool(g.device, &cpci, nullptr, &pool));
    VkCommandBufferAllocateInfo cbai = {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = nullptr;
    VK_CHECK(vkAllocateCommandBuffers(g.device, &cbai, &cmd));
    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));
    VkBufferCopy region = {};
    region.size = bytes;
    vkCmdCopyBuffer(cmd, staging.handle, dst.handle, 1, &region);
    VK_CHECK(vkEndCommandBuffer(cmd));
    VkFenceCreateInfo fci = {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = nullptr;
    VK_CHECK(vkCreateFence(g.device, &fci, nullptr, &fence));
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(g.queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(g.device, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(g.device, fence, nullptr);
    vkDestroyCommandPool(g.device, pool, nullptr);
    vkFreeMemory(g.device, staging.memory, nullptr);
    vkDestroyBuffer(g.device, staging.handle, nullptr);
}

void free_buffer(const Gpu &g, Buffer &b)
{
    vkFreeMemory(g.device, b.memory, nullptr);
    vkDestroyBuffer(g.device, b.handle, nullptr);
}

void upload_floats(const Gpu &g, const Buffer &b, const std::vector<float> &data)
{
    void *mapped = nullptr;
    VK_CHECK(vkMapMemory(g.device, b.memory, 0, b.size, 0, &mapped));
    std::memcpy(mapped, data.data(), data.size() * sizeof(float));
    vkUnmapMemory(g.device, b.memory);
}

VkShaderModule load_shader(const Gpu &g, const std::filesystem::path &path)
{
    std::vector<char> spirv = read_file(path);
    VkShaderModuleCreateInfo smci = {};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spirv.size();
    smci.pCode = reinterpret_cast<const uint32_t *>(spirv.data());
    VkShaderModule module = nullptr;
    VK_CHECK(vkCreateShaderModule(g.device, &smci, nullptr, &module));
    return module;
}

// Records, submits, and waits for a single dispatch of an already-bound
// pipeline, then barriers the frame buffer for host read.
// Records, submits, and waits for a single dispatch. When readback is set,
// the frame is copied to that host-visible staging buffer in the same
// submit (shader-write -> transfer-read, copy, transfer-write -> host-read),
// so one fence covers everything. Timestamps bracket the dispatch alone:
// copy traffic is a separate concern from shader work.
void dispatch_and_wait(const Gpu &g, VkPipeline pipeline, VkPipelineLayout layout,
                       VkDescriptorSet set, const void *push_data, size_t push_size,
                       const Buffer &frame, uint32_t gx, uint32_t gy,
                       const Buffer *readback = nullptr)
{
    VkCommandPoolCreateInfo cpci = {};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = g.qfamily;
    VkCommandPool cmd_pool = nullptr;
    VK_CHECK(vkCreateCommandPool(g.device, &cpci, nullptr, &cmd_pool));
    VkCommandBufferAllocateInfo cbai = {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cmd_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = nullptr;
    VK_CHECK(vkAllocateCommandBuffers(g.device, &cbai, &cmd));
    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));
    if (readback)
    {
        // Prior uploads rode separate submits on this queue. Submission order
        // executes in order; this barrier adds the memory-visibility half:
        // transfer writes available before any shader read below. Fill/normal
        // skip it — their buffers were never transferred.
        VkMemoryBarrier up = {};
        up.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        up.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        up.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &up, 0, nullptr, 0, nullptr);
    }
    // Two timestamp queries bracket the dispatch when the queue supports
    // them; the pool is per-dispatch on purpose (bench tool, not a hot loop).
    VkQueryPool query_pool = nullptr;
    bool timed = g.timestamp_period > 0.0f;
    if (timed)
    {
        VkQueryPoolCreateInfo qpci = {};
        qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2;
        VK_CHECK(vkCreateQueryPool(g.device, &qpci, nullptr, &query_pool));
        vkCmdResetQueryPool(cmd, query_pool, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, 0);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       static_cast<uint32_t>(push_size), push_data);
    vkCmdDispatch(cmd, gx, gy, 1);
    if (timed)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, 1);
    if (readback)
    {
        VkBufferMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.srcQueueFamilyIndex = g.qfamily;
        barrier.dstQueueFamilyIndex = g.qfamily;
        barrier.buffer = frame.handle;
        barrier.size = frame.size;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 1, &barrier, 0, nullptr);
        VkBufferCopy region = {};
        region.size = frame.size;
        vkCmdCopyBuffer(cmd, frame.handle, readback->handle, 1, &region);
        VkBufferMemoryBarrier host_barrier = barrier;
        host_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        host_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        host_barrier.buffer = readback->handle;
        host_barrier.size = readback->size;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 0, nullptr, 1, &host_barrier, 0, nullptr);
        VK_CHECK(vkEndCommandBuffer(cmd));
    }
    else
    {
    VkBufferMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    barrier.srcQueueFamilyIndex = g.qfamily;
    barrier.dstQueueFamilyIndex = g.qfamily;
    barrier.buffer = frame.handle;
    barrier.size = frame.size;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 0, nullptr, 1, &barrier, 0, nullptr);
    VK_CHECK(vkEndCommandBuffer(cmd));
    }
    VkFenceCreateInfo fci = {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = nullptr;
    VK_CHECK(vkCreateFence(g.device, &fci, nullptr, &fence));
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(g.queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(g.device, 1, &fence, VK_TRUE, UINT64_MAX));
    if (timed)
    {
        uint64_t ticks[2] = {};
        VK_CHECK(vkGetQueryPoolResults(g.device, query_pool, 0, 2, sizeof(ticks), ticks,
                                       sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        double ms = (ticks[1] - ticks[0]) * g.timestamp_period / 1e6;
        std::printf("[gpu] dispatch=%.3fms\n", ms);
        vkDestroyQueryPool(g.device, query_pool, nullptr);
    }
    vkDestroyFence(g.device, fence, nullptr);
    vkDestroyCommandPool(g.device, cmd_pool, nullptr);
}

VkDescriptorSetLayout make_layout(const Gpu &g, uint32_t bindings)
{
    std::vector<VkDescriptorSetLayoutBinding> b(bindings);
    for (uint32_t i = 0; i < bindings; ++i)
    {
        b[i].binding = i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsli = {};
    dsli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsli.bindingCount = bindings;
    dsli.pBindings = b.data();
    VkDescriptorSetLayout layout = nullptr;
    VK_CHECK(vkCreateDescriptorSetLayout(g.device, &dsli, nullptr, &layout));
    return layout;
}

VkPipeline make_pipeline(const Gpu &g, VkShaderModule module, VkPipelineLayout layout,
                         const VkSpecializationInfo *spec = nullptr)
{
    VkPipelineShaderStageCreateInfo stage = {};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";
    stage.pSpecializationInfo = spec;
    VkComputePipelineCreateInfo pci = {};
    pci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pci.stage = stage;
    pci.layout = layout;
    VkPipeline pipeline = nullptr;
    VK_CHECK(vkCreateComputePipelines(g.device, nullptr, 1, &pci, nullptr, &pipeline));
    return pipeline;
}

// Binds an array of storage buffers to consecutive bindings of one set.
VkDescriptorSet bind_buffers(const Gpu &g, VkDescriptorSetLayout layout,
                             const std::vector<Buffer> &buffers)
{
    VkDescriptorPoolSize pool_size = {};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = static_cast<uint32_t>(buffers.size());
    VkDescriptorPoolCreateInfo dpi = {};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 1;
    dpi.pPoolSizes = &pool_size;
    dpi.poolSizeCount = 1;
    VkDescriptorPool pool = nullptr;
    VK_CHECK(vkCreateDescriptorPool(g.device, &dpi, nullptr, &pool));
    VkDescriptorSetAllocateInfo dsai = {};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &layout;
    VkDescriptorSet set = nullptr;
    VK_CHECK(vkAllocateDescriptorSets(g.device, &dsai, &set));
    std::vector<VkDescriptorBufferInfo> infos(buffers.size());
    std::vector<VkWriteDescriptorSet> writes(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i)
    {
        infos[i].buffer = buffers[i].handle;
        infos[i].range = buffers[i].size;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(g.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    // Pool intentionally leaked until device teardown: one set per run, and
    // the pool dies with the device. A long-lived app would retain and reset it.
    (void)pool;
    return set;
}

std::vector<vec3> download_frame(const Gpu &g, const Buffer &frame, uint32_t width, uint32_t height)
{
    void *mapped = nullptr;
    VK_CHECK(vkMapMemory(g.device, frame.memory, 0, frame.size, 0, &mapped));
    const float *pixels = static_cast<const float *>(mapped);
    std::vector<vec3> fb(static_cast<size_t>(width) * height);
    for (size_t i = 0; i < fb.size(); ++i)
        fb[i] = vec3(pixels[i * 3], pixels[i * 3 + 1], pixels[i * 3 + 2]);
    vkUnmapMemory(g.device, frame.memory);
    return fb;
}

int run_fill(Gpu &g, const std::filesystem::path &shader_dir)
{
    const uint32_t width = 800;
    const uint32_t height = 450;

    Buffer frame = make_mapped_buffer(g, static_cast<VkDeviceSize>(width) * height * 3 * sizeof(float));

    VkDescriptorSetLayout layout = make_layout(g, 1);
    VkPushConstantRange push = {};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.size = 8;
    VkPipelineLayoutCreateInfo pli = {};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &push;
    VkPipelineLayout pipeline_layout = nullptr;
    VK_CHECK(vkCreatePipelineLayout(g.device, &pli, nullptr, &pipeline_layout));

    VkDescriptorSet set = bind_buffers(g, layout, {frame});
    VkShaderModule module = load_shader(g, shader_dir / "fill.spv");
    VkPipeline pipeline = make_pipeline(g, module, pipeline_layout);

    uint32_t dims[2] = {width, height};
    dispatch_and_wait(g, pipeline, pipeline_layout, set, dims, sizeof(dims), frame,
                      (width + 15) / 16, (height + 15) / 16);

    std::vector<vec3> fb = download_frame(g, frame, width, height);
    std::uint64_t hash = write_ppm("gpu_fill.ppm", fb, width, height, 1.0, true);
    std::printf("wrote gpu_fill.ppm hash=%llu\n", static_cast<unsigned long long>(hash));

    vkDestroyPipeline(g.device, pipeline, nullptr);
    vkDestroyShaderModule(g.device, module, nullptr);
    vkDestroyPipelineLayout(g.device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(g.device, layout, nullptr);
    free_buffer(g, frame);
    return 0;
}

// Push block mirror of normal.comp's Push (std430: vec4 slots, then words).
struct NormalPush
{
    float origin[4];
    float llc[4];
    float horiz[4];
    float vert[4];
    uint32_t width, height, n_spheres, n_tris;
    uint32_t n_quads, pad0, pad1, pad2;
};

// Flattened material: rgb carries solid albedo or emission, w carries the
// dielectric IOR; kind selects the branch, texkind/texidx select the texture
// (0 none, 1 checker, 2 image).
struct FlatMat
{
    vec3 rgb = vec3(0.5, 0.5, 0.5);
    double w = 0.0;
    int kind = 0;
    int texkind = 0;
    int texidx = -1;
};

struct TexCollectors
{
    std::vector<const checker_texture *> checkers;
    std::vector<const image_texture *> images;
};

inline int find_or_add_checker(TexCollectors &tc, const checker_texture *c)
{
    for (size_t i = 0; i < tc.checkers.size(); ++i)
        if (tc.checkers[i] == c)
            return static_cast<int>(i);
    tc.checkers.push_back(c);
    return static_cast<int>(tc.checkers.size() - 1);
}

inline int find_or_add_image(TexCollectors &tc, const image_texture *im)
{
    for (size_t i = 0; i < tc.images.size(); ++i)
        if (tc.images[i] == im)
            return static_cast<int>(i);
    tc.images.push_back(im);
    return static_cast<int>(tc.images.size() - 1);
}

inline FlatMat flatten_material(const std::shared_ptr<material> &m, TexCollectors &tc)
{
    FlatMat f;
    if (const auto *l = dynamic_cast<const lambertian *>(m.get()))
    {
        const auto &t = l->tex();
        if (const auto *s = dynamic_cast<const solid_color *>(t.get()))
        {
            f.rgb = s->color();
            return f; // kind 0, no texture
        }
        if (const auto *c = dynamic_cast<const checker_texture *>(t.get()))
        {
            f.texkind = 1;
            f.texidx = find_or_add_checker(tc, c);
            return f;
        }
        if (const auto *im = dynamic_cast<const image_texture *>(t.get()))
        {
            f.texkind = 2;
            f.texidx = find_or_add_image(tc, im);
            return f;
        }
        std::fprintf(stderr, "upload: unknown texture type, gray fallback\n");
        return f;
    }
    if (const auto *me = dynamic_cast<const metal *>(m.get()))
    {
        f.rgb = me->tint();
        f.kind = 1;
        return f;
    }
    if (const auto *d = dynamic_cast<const dielectric *>(m.get()))
    {
        f.rgb = vec3(1, 1, 1);
        f.w = d->index();
        f.kind = 2;
        return f;
    }
    if (const auto *li = dynamic_cast<const diffuse_light *>(m.get()))
    {
        f.rgb = li->emission();
        f.kind = 3;
        return f;
    }
    std::fprintf(stderr, "upload: unknown material type, gray fallback\n");
    return f;
}

int run_normal(Gpu &g, const std::filesystem::path &shader_dir)
{
    const uint32_t width = 800;
    const uint32_t height = 450;

    // Same scene, same seed as the CPU parity run
    // (--bench --samples 1 --shade normal --nee --aperture 0).
    render_config cfg;
    cfg.bench = true;
    cfg.extra_spheres = 300;
    cfg.do_nee = true;
    set_deterministic_rng(true, cfg.bench_seed);
    scene_data scene = build_scene(cfg);
    camera cam = default_camera(0.0);

    // Flatten to SoA: spheres as center+radius, tris/quads as corner+edges.
    std::vector<float> sph, tri, qd;
    auto push_vec3 = [](std::vector<float> &v, const vec3 &p, float w) {
        v.push_back(static_cast<float>(p.x()));
        v.push_back(static_cast<float>(p.y()));
        v.push_back(static_cast<float>(p.z()));
        v.push_back(w);
    };
    for (const auto &o : scene.objects.objects_ref())
    {
        if (const auto *s = dynamic_cast<const sphere *>(o.get()))
            push_vec3(sph, s->position(), static_cast<float>(s->size()));
        else if (const auto *t = dynamic_cast<const triangle *>(o.get()))
        {
            push_vec3(tri, t->a(), 0.0f);
            push_vec3(tri, t->b(), 0.0f);
            push_vec3(tri, t->c(), 0.0f);
        }
        else if (const auto *q = dynamic_cast<const quad *>(o.get()))
        {
            push_vec3(qd, q->corner(), 0.0f);
            push_vec3(qd, q->edge_u(), 0.0f);
            push_vec3(qd, q->edge_v(), 0.0f);
        }
    }
    uint32_t n_spheres = static_cast<uint32_t>(sph.size() / 4);
    uint32_t n_tris = static_cast<uint32_t>(tri.size() / 12);
    uint32_t n_quads = static_cast<uint32_t>(qd.size() / 12);
    std::printf("upload: %u spheres %u tris %u quads\n", n_spheres, n_tris, n_quads);
    // Zero-size Vulkan buffers are invalid; scene always has all three kinds,
    // the guard is against future empty scenes, not today's.
    if (sph.empty())
        sph.resize(4, 0.0f);
    if (tri.empty())
        tri.resize(12, 0.0f);
    if (qd.empty())
        qd.resize(12, 0.0f);

    Buffer b_sph = make_mapped_buffer(g, sph.size() * sizeof(float));
    Buffer b_tri = make_mapped_buffer(g, tri.size() * sizeof(float));
    Buffer b_qd = make_mapped_buffer(g, qd.size() * sizeof(float));
    Buffer frame = make_mapped_buffer(g, static_cast<VkDeviceSize>(width) * height * 3 * sizeof(float));
    upload_floats(g, b_sph, sph);
    upload_floats(g, b_tri, tri);
    upload_floats(g, b_qd, qd);

    VkDescriptorSetLayout layout = make_layout(g, 4);
    VkPushConstantRange push = {};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.size = sizeof(NormalPush);
    VkPipelineLayoutCreateInfo pli = {};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &push;
    VkPipelineLayout pipeline_layout = nullptr;
    VK_CHECK(vkCreatePipelineLayout(g.device, &pli, nullptr, &pipeline_layout));

    VkDescriptorSet set = bind_buffers(g, layout, {b_sph, b_tri, b_qd, frame});
    VkShaderModule module = load_shader(g, shader_dir / "normal.spv");
    VkPipeline pipeline = make_pipeline(g, module, pipeline_layout);

    NormalPush pc = {};
    auto fill_v4 = [](float *d, const vec3 &v) {
        d[0] = static_cast<float>(v.x());
        d[1] = static_cast<float>(v.y());
        d[2] = static_cast<float>(v.z());
        d[3] = 0.0f;
    };
    fill_v4(pc.origin, cam.eye());
    fill_v4(pc.llc, cam.corner());
    fill_v4(pc.horiz, cam.span_h());
    fill_v4(pc.vert, cam.span_v());
    pc.width = width;
    pc.height = height;
    pc.n_spheres = n_spheres;
    pc.n_tris = n_tris;
    pc.n_quads = n_quads;

    dispatch_and_wait(g, pipeline, pipeline_layout, set, &pc, sizeof(pc), frame,
                      (width + 15) / 16, (height + 15) / 16);

    std::vector<vec3> fb = download_frame(g, frame, width, height);
    std::uint64_t hash = write_ppm("gpu_normal.ppm", fb, width, height, 1.0, true);
    std::printf("wrote gpu_normal.ppm hash=%llu\n", static_cast<unsigned long long>(hash));

    vkDestroyPipeline(g.device, pipeline, nullptr);
    vkDestroyShaderModule(g.device, module, nullptr);
    vkDestroyPipelineLayout(g.device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(g.device, layout, nullptr);
    free_buffer(g, frame);
    free_buffer(g, b_qd);
    free_buffer(g, b_tri);
    free_buffer(g, b_sph);
    return 0;
}

// Push block mirror of path.comp's Push.
struct PathPush
{
    float origin[4];
    float llc[4];
    float horiz[4];
    float vert[4];
    uint32_t width, height, n_spheres, n_tris;
    uint32_t n_quads, samples, strat_n, max_depth;
    uint32_t img_w, img_h, use_bvh, probe;
};

int run_path(Gpu &g, const std::filesystem::path &shader_dir, int samples, bool use_bvh, int extra_spheres,
             bool use_glass, uint32_t wg_x, uint32_t wg_y, bool probe)
{
    int strat_n = static_cast<int>(std::sqrt(samples + 0.5));
    if (strat_n * strat_n != samples || strat_n <= 0)
    {
        std::fprintf(stderr, "path samples must be a perfect square (stratified mapping)\n");
        return 1;
    }
    const uint32_t width = 800;
    const uint32_t height = 450;
    const int max_depth = 50;

    // Same scene, same seed as the CPU parity run (--bench --nee).
    render_config cfg;
    cfg.bench = true;
    cfg.extra_spheres = extra_spheres;
    cfg.do_nee = true;
    cfg.use_glass = use_glass;
    set_deterministic_rng(true, cfg.bench_seed);
    scene_data scene = build_scene(cfg);
    camera cam = default_camera(0.05);

    // Flatten prims + materials together so indices line up across buffers,
    // recording each prim's (type,index) for the BVH leaf refs below.
    std::vector<float> sph, tri, qd;
    std::vector<float> sph_alb, tri_alb, qd_alb;
    std::vector<int32_t> sph_meta, tri_meta, qd_meta;
    std::map<const hittable *, FlatLeafRef> prim_ids;
    TexCollectors tc;
    auto push_vec3 = [](std::vector<float> &v, const vec3 &p, float w) {
        v.push_back(static_cast<float>(p.x()));
        v.push_back(static_cast<float>(p.y()));
        v.push_back(static_cast<float>(p.z()));
        v.push_back(w);
    };
    auto push_mat = [&](std::vector<float> &va, std::vector<int32_t> &vm,
                        const std::shared_ptr<material> &m) {
        FlatMat f = flatten_material(m, tc);
        push_vec3(va, f.rgb, static_cast<float>(f.w));
        vm.push_back(f.kind);
        vm.push_back(f.texkind);
        vm.push_back(f.texidx);
        vm.push_back(0);
    };
    for (const auto &o : scene.objects.objects_ref())
    {
        if (const auto *s = dynamic_cast<const sphere *>(o.get()))
        {
            prim_ids[o.get()] = FlatLeafRef{0, static_cast<int>(sph.size() / 4)};
            push_vec3(sph, s->position(), static_cast<float>(s->size()));
            push_mat(sph_alb, sph_meta, s->mat_ptr());
        }
        else if (const auto *t = dynamic_cast<const triangle *>(o.get()))
        {
            prim_ids[o.get()] = FlatLeafRef{1, static_cast<int>(tri.size() / 12)};
            push_vec3(tri, t->a(), 0.0f);
            push_vec3(tri, t->b(), 0.0f);
            push_vec3(tri, t->c(), 0.0f);
            push_mat(tri_alb, tri_meta, t->mat_ptr());
        }
        else if (const auto *q = dynamic_cast<const quad *>(o.get()))
        {
            prim_ids[o.get()] = FlatLeafRef{2, static_cast<int>(qd.size() / 12)};
            push_vec3(qd, q->corner(), 0.0f);
            push_vec3(qd, q->edge_u(), 0.0f);
            push_vec3(qd, q->edge_v(), 0.0f);
            push_mat(qd_alb, qd_meta, q->mat_ptr());
        }
    }
    uint32_t n_spheres = static_cast<uint32_t>(sph.size() / 4);
    uint32_t n_tris = static_cast<uint32_t>(tri.size() / 12);
    uint32_t n_quads = static_cast<uint32_t>(qd.size() / 12);
    std::printf("upload: %u spheres %u tris %u quads, %zu checkers, %zu images\n",
                n_spheres, n_tris, n_quads, tc.checkers.size(), tc.images.size());

    // Same SAH tree the CPU traces (leaf size 2 matches the bench default).
    // Node links occupy ivec4 slots [0, nnodes), leaf refs follow them, so
    // leaf starts are absolute buffer indices — no pointer fixups.
    bvh_node bvh_root(scene.objects, 2);
    std::vector<FlatNode> fnodes;
    std::vector<FlatLeafRef> frefs;
    bvh_root.flatten(fnodes, frefs, [&](const std::shared_ptr<hittable> &p) { return prim_ids[p.get()]; });
    std::printf("bvh: %zu nodes, %zu leaf refs\n", fnodes.size(), frefs.size());
    std::vector<float> bvh_box;
    bvh_box.reserve(fnodes.size() * 8);
    std::vector<int32_t> bvh_link;
    bvh_link.reserve((fnodes.size() + frefs.size()) * 4);
    for (const auto &n : fnodes)
    {
        bvh_box.push_back(static_cast<float>(n.bmin.x()));
        bvh_box.push_back(static_cast<float>(n.bmin.y()));
        bvh_box.push_back(static_cast<float>(n.bmin.z()));
        bvh_box.push_back(0.0f);
        bvh_box.push_back(static_cast<float>(n.bmax.x()));
        bvh_box.push_back(static_cast<float>(n.bmax.y()));
        bvh_box.push_back(static_cast<float>(n.bmax.z()));
        bvh_box.push_back(0.0f);
        bvh_link.push_back(n.left);
        bvh_link.push_back(n.right);
        bvh_link.push_back(n.left < 0 ? static_cast<int32_t>(fnodes.size()) + n.start : 0);
        bvh_link.push_back(n.count);
    }
    for (const auto &r : frefs)
    {
        bvh_link.push_back(r.type);
        bvh_link.push_back(r.index);
        bvh_link.push_back(0);
        bvh_link.push_back(0);
    }

    // Checker table: (scale,0,0,0),(c1,0),(c2,0) per entry.
    std::vector<float> chk;
    for (const auto *c : tc.checkers)
    {
        auto solid_rgb = [](const std::shared_ptr<texture> &t) {
            if (const auto *s = dynamic_cast<const solid_color *>(t.get()))
                return s->color();
            return vec3(0.5, 0.5, 0.5);
        };
        push_vec3(chk, vec3(c->scale(), 0, 0), 0.0f);
        push_vec3(chk, solid_rgb(c->color_a()), 0.0f);
        push_vec3(chk, solid_rgb(c->color_b()), 0.0f);
    }
    if (chk.empty())
        chk.resize(12, 0.0f);

    // Image textures: raw bytes as floats (shader divides by 255 like CPU).
    // One image slot for now; the scene holds exactly one.
    std::vector<float> img;
    uint32_t img_w = 1, img_h = 1;
    if (!tc.images.empty())
    {
        if (tc.images.size() > 1)
            std::fprintf(stderr, "upload: %zu images, using the first\n", tc.images.size());
        const auto *im = tc.images[0];
        img_w = static_cast<uint32_t>(im->pixel_width());
        img_h = static_cast<uint32_t>(im->pixel_height());
        img.reserve(static_cast<size_t>(img_w) * img_h * 3);
        for (unsigned char b : im->bytes())
            img.push_back(static_cast<float>(b));
    }
    else
    {
        img.resize(3, 128.0f);
    }

    if (sph.empty())
        sph.resize(4, 0.0f);
    if (tri.empty())
        tri.resize(12, 0.0f);
    if (qd.empty())
        qd.resize(12, 0.0f);

    // Device-local scene buffers fed by staging uploads; the framebuffer is
    // device-local too, read back through a staging copy in the dispatch.
    const VkMemoryPropertyFlags dev_props =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    auto device_floats = [&](const std::vector<float> &v) {
        Buffer b = make_buffer(g, v.size() * sizeof(float),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               dev_props);
        upload_to_device(g, b, v.data(), v.size() * sizeof(float));
        return b;
    };
    auto device_ints = [&](const std::vector<int32_t> &v) {
        Buffer b = make_buffer(g, v.size() * sizeof(int32_t),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               dev_props);
        upload_to_device(g, b, v.data(), v.size() * sizeof(int32_t));
        return b;
    };
    Buffer b_sph = device_floats(sph);
    Buffer b_tri = device_floats(tri);
    Buffer b_qd = device_floats(qd);
    Buffer b_sph_alb = device_floats(sph_alb);
    Buffer b_sph_meta = device_ints(sph_meta);
    Buffer b_tri_alb = device_floats(tri_alb);
    Buffer b_tri_meta = device_ints(tri_meta);
    Buffer b_qd_alb = device_floats(qd_alb);
    Buffer b_qd_meta = device_ints(qd_meta);
    Buffer b_chk = device_floats(chk);
    Buffer b_img = device_floats(img);
    Buffer b_bvh_box = device_floats(bvh_box);
    Buffer b_bvh_link = device_ints(bvh_link);
    const VkDeviceSize frame_bytes = static_cast<VkDeviceSize>(width) * height * 3 * sizeof(float);
    Buffer frame = make_buffer(g, frame_bytes,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               dev_props);
    Buffer staging = make_buffer(g, frame_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    // Probe buffer: per-pixel segment totals for the divergence histogram.
    // Always bound (statically used), written only in probe mode.
    const VkDeviceSize probe_bytes = static_cast<VkDeviceSize>(width) * height * sizeof(uint32_t);
    Buffer probe_buf = make_buffer(g, probe_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    {
        void *zero = nullptr;
        VK_CHECK(vkMapMemory(g.device, probe_buf.memory, 0, probe_bytes, 0, &zero));
        std::memset(zero, 0, static_cast<size_t>(probe_bytes));
        vkUnmapMemory(g.device, probe_buf.memory);
    }

    VkDescriptorSetLayout layout = make_layout(g, 15);
    VkPushConstantRange push = {};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.size = sizeof(PathPush);
    VkPipelineLayoutCreateInfo pli = {};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &push;
    VkPipelineLayout pipeline_layout = nullptr;
    VK_CHECK(vkCreatePipelineLayout(g.device, &pli, nullptr, &pipeline_layout));

    VkDescriptorSet set = bind_buffers(g, layout, {b_sph, b_tri, b_qd, b_sph_alb, b_sph_meta,
                                                   b_tri_alb, b_tri_meta, b_qd_alb, b_qd_meta,
                                                   b_chk, b_img, frame, b_bvh_box, b_bvh_link,
                                                   probe_buf});
    VkShaderModule module = load_shader(g, shader_dir / "path.spv");
    // Workgroup size resolves the shader's specialization constants: same
    // SPIR-V for every occupancy experiment below.
    uint32_t wg_vals[2] = {wg_x, wg_y};
    VkSpecializationMapEntry wg_entries[2] = {};
    wg_entries[0].constantID = 0;
    wg_entries[0].size = sizeof(uint32_t);
    wg_entries[1].constantID = 1;
    wg_entries[1].offset = sizeof(uint32_t);
    wg_entries[1].size = sizeof(uint32_t);
    VkSpecializationInfo wg_spec = {};
    wg_spec.mapEntryCount = 2;
    wg_spec.pMapEntries = wg_entries;
    wg_spec.dataSize = sizeof(wg_vals);
    wg_spec.pData = wg_vals;
    VkPipeline pipeline = make_pipeline(g, module, pipeline_layout, &wg_spec);

    PathPush pc = {};
    auto fill_v4 = [](float *d, const vec3 &v) {
        d[0] = static_cast<float>(v.x());
        d[1] = static_cast<float>(v.y());
        d[2] = static_cast<float>(v.z());
        d[3] = 0.0f;
    };
    fill_v4(pc.origin, cam.eye());
    fill_v4(pc.llc, cam.corner());
    fill_v4(pc.horiz, cam.span_h());
    fill_v4(pc.vert, cam.span_v());
    pc.width = width;
    pc.height = height;
    pc.n_spheres = n_spheres;
    pc.n_tris = n_tris;
    pc.n_quads = n_quads;
    pc.samples = static_cast<uint32_t>(samples);
    pc.strat_n = static_cast<uint32_t>(strat_n);
    pc.max_depth = static_cast<uint32_t>(max_depth);
    pc.img_w = img_w;
    pc.img_h = img_h;
    pc.use_bvh = use_bvh ? 1u : 0u;
    pc.probe = probe ? 1u : 0u;

    dispatch_and_wait(g, pipeline, pipeline_layout, set, &pc, sizeof(pc), frame,
                      (width + wg_x - 1) / wg_x, (height + wg_y - 1) / wg_y, &staging);

    std::vector<vec3> fb = download_frame(g, staging, width, height);
    std::string out_name = use_bvh ? "gpu_path.ppm" : "gpu_path_flat.ppm";
    if (probe)
        out_name = "gpu_probe.ppm";
    std::uint64_t hash = write_ppm(out_name, fb, width, height, 1.0, true);
    std::printf("wrote %s hash=%llu\n", out_name.c_str(), static_cast<unsigned long long>(hash));

    if (probe)
    {
        // Path-length histogram: segments per pixel across all samples.
        // Wide spread = threads in a warp retire at wildly different times =
        // what wavefront compaction would fix. Tight = megakernel is fine.
        void *mapped = nullptr;
        VK_CHECK(vkMapMemory(g.device, probe_buf.memory, 0, probe_bytes, 0, &mapped));
        const uint32_t *counts = static_cast<const uint32_t *>(mapped);
        size_t total_px = static_cast<size_t>(width) * height;
        std::vector<size_t> hist(64, 0);
        size_t over = 0;
        uint64_t sum = 0;
        uint32_t mx = 0;
        for (size_t i = 0; i < total_px; ++i)
        {
            uint32_t c = counts[i];
            sum += c;
            if (c > mx)
                mx = c;
            // Bucket by mean segments per sample (0..63, overflow last).
            size_t b = (c / static_cast<uint32_t>(samples >= 1 ? samples : 1));
            if (b > 63)
            {
                b = 63;
                ++over;
            }
            hist[b]++;
        }
        vkUnmapMemory(g.device, probe_buf.memory);
        double mean = static_cast<double>(sum) / total_px / samples;
        std::printf("[probe] mean_segs_per_sample=%.2f max_total=%u\n", mean, mx);
        std::printf("[probe] buckets(mean segs : pixels):");
        for (size_t b = 0; b < 64; ++b)
            if (hist[b] > 0)
                std::printf(" %zu:%zu", b, hist[b]);
        std::printf(" over63:%zu\n", over);
    }

    vkDestroyPipeline(g.device, pipeline, nullptr);
    vkDestroyShaderModule(g.device, module, nullptr);
    vkDestroyPipelineLayout(g.device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(g.device, layout, nullptr);
    free_buffer(g, staging);
    free_buffer(g, probe_buf);
    free_buffer(g, frame);
    free_buffer(g, b_bvh_link);
    free_buffer(g, b_bvh_box);
    free_buffer(g, b_img);
    free_buffer(g, b_chk);
    free_buffer(g, b_qd_meta);
    free_buffer(g, b_qd_alb);
    free_buffer(g, b_tri_meta);
    free_buffer(g, b_tri_alb);
    free_buffer(g, b_sph_meta);
    free_buffer(g, b_sph_alb);
    free_buffer(g, b_qd);
    free_buffer(g, b_tri);
    free_buffer(g, b_sph);
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    Gpu g = init_gpu();
    std::filesystem::path shader_dir = exe_dir(argv[0]) / "shaders";
    std::string mode = (argc > 1) ? argv[1] : "fill";
    int rc = 0;
    if (mode == "normal")
        rc = run_normal(g, shader_dir);
    else if (mode == "path" || mode == "probe")
    {
        // path [samples] [flat|bvh] [spheres] [glass] [WxH]: traversal A/B
        // plus workgroup occupancy in one binary. probe [samples] [spheres]:
        // same render plus a per-pixel path-length histogram for the
        // divergence verdict (stdout, no image comparison needed).
        int samples = (argc > 2) ? std::atoi(argv[2]) : 196;
        bool use_bvh = (argc <= 3) || (std::string(argv[3]) != "flat");
        int spheres = (argc > 4) ? std::atoi(argv[4]) : 300;
        bool glass = (argc > 5) && (std::string(argv[5]) == "glass");
        // Default 16x8 measured ~11% faster than 8x8 on this GPU (128 threads
        // fill better; 256-thread groups lose to register pressure).
        uint32_t wg_x = 16, wg_y = 8;
        if (argc > 6)
        {
            std::string wg = argv[6];
            size_t x = wg.find('x');
            if (x != std::string::npos)
            {
                wg_x = static_cast<uint32_t>(std::atoi(wg.substr(0, x).c_str()));
                wg_y = static_cast<uint32_t>(std::atoi(wg.substr(x + 1).c_str()));
            }
            if (wg_x == 0 || wg_y == 0 || wg_x * wg_y > 1024)
            {
                std::fprintf(stderr, "bad workgroup size (try 8x8)\n");
                return 1;
            }
        }
        if (mode == "probe")
        {
            // Probe defaults mirror the parity scene; explicit flags win.
            if (argc <= 2)
                samples = 64;
            if (argc <= 4)
                spheres = 300;
            use_bvh = true;
        }
        rc = run_path(g, shader_dir, samples, use_bvh, spheres, glass, wg_x, wg_y, mode == "probe");
    }
    else
        rc = run_fill(g, shader_dir);
    shutdown_gpu(g);
    return rc;
}
