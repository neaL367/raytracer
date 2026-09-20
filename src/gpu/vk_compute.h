#pragma once
// Headless Vulkan compute runner. Scene-agnostic: bytes in (buffers),
// floats out (RGBA image). Knows nothing about spheres or materials.
#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define VKC_CHECK(x)                                                     \
    do {                                                                 \
        VkResult r = (x);                                                \
        if (r != VK_SUCCESS) {                                           \
            std::cerr << "vulkan error " << r << " at " << __LINE__      \
                      << "\n";                                           \
            std::exit(1);                                                \
        }                                                                \
    } while (0)

struct GpuBuffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    size_t bytes = 0;
};

struct GpuContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t qfam = 0;
    int W = 0, H = 0;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory image_mem = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    GpuBuffer staging;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkQueryPool query_pool = VK_NULL_HANDLE;
    GpuBuffer scene_bufs[6];
    bool has_scene = false;
};

inline uint32_t vkc_find_memory(VkPhysicalDevice pd, uint32_t bits,
                                VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(pd, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & want) == want)
            return i;
    std::cerr << "no suitable memory type\n";
    std::exit(1);
}

// Instance + discrete-NVIDIA-first device + image/staging/layout/pool.
inline void gpu_init(GpuContext &g, int W, int H) {
    g.W = W;
    g.H = H;
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "rt_gpu";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VKC_CHECK(vkCreateInstance(&ici, nullptr, &g.instance));

    {
        uint32_t n = 0;
        VKC_CHECK(vkEnumeratePhysicalDevices(g.instance, &n, nullptr));
        std::vector<VkPhysicalDevice> pds(n);
        VKC_CHECK(vkEnumeratePhysicalDevices(g.instance, &n, pds.data()));
        int best = -1;
        for (auto pd : pds) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(pd, &props);
            uint32_t qn = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
            std::vector<VkQueueFamilyProperties> qs(qn);
            vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qs.data());
            int qfam = -1;
            for (uint32_t i = 0; i < qn; ++i)
                if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    qfam = (int)i;
                    break;
                }
            if (qfam < 0)
                continue;
            int score = 0;
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
                score += 1000;
            if (props.vendorID == 0x10DE)
                score += 500;
            if (score > best) {
                best = score;
                g.phys = pd;
                g.qfam = (uint32_t)qfam;
            }
        }
        if (g.phys == VK_NULL_HANDLE) {
            std::cerr << "no compute device\n";
            std::exit(1);
        }
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(g.phys, &props);
        std::cout << "gpu: " << props.deviceName << "\n";
    }

    {
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qi{};
        qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = g.qfam;
        qi.queueCount = 1;
        qi.pQueuePriorities = &prio;
        VkDeviceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.queueCreateInfoCount = 1;
        ci.pQueueCreateInfos = &qi;
        VKC_CHECK(vkCreateDevice(g.phys, &ci, nullptr, &g.device));
        vkGetDeviceQueue(g.device, g.qfam, 0, &g.queue);
    }

    {
        VkImageCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        ci.extent = {(uint32_t)W, (uint32_t)H, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VKC_CHECK(vkCreateImage(g.device, &ci, nullptr, &g.image));
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(g.device, g.image, &req);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = vkc_find_memory(g.phys, req.memoryTypeBits,
                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VKC_CHECK(vkAllocateMemory(g.device, &ai, nullptr, &g.image_mem));
        VKC_CHECK(vkBindImageMemory(g.device, g.image, g.image_mem, 0));
    }
    {
        VkImageViewCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image = g.image;
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VKC_CHECK(vkCreateImageView(g.device, &ci, nullptr, &g.view));
    }
    {
        VkBufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size = (size_t)W * H * 16;
        ci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VKC_CHECK(vkCreateBuffer(g.device, &ci, nullptr, &g.staging.buf));
        g.staging.bytes = ci.size;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(g.device, g.staging.buf, &req);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex =
            vkc_find_memory(g.phys, req.memoryTypeBits,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VKC_CHECK(vkAllocateMemory(g.device, &ai, nullptr, &g.staging.mem));
        VKC_CHECK(vkBindBufferMemory(g.device, g.staging.buf, g.staging.mem, 0));
    }
    {
        VkDescriptorSetLayoutBinding b[7]{};
        b[0].binding = 0;
        b[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[0].descriptorCount = 1;
        b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        for (uint32_t i = 1; i < 7; ++i) {
            b[i].binding = i;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            b[i].descriptorType = (i == 1) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                           : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 7;
        ci.pBindings = b;
        VKC_CHECK(vkCreateDescriptorSetLayout(g.device, &ci, nullptr, &g.layout));
        VkDescriptorPoolSize ps[3]{};
        ps[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        ps[0].descriptorCount = 1;
        ps[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ps[1].descriptorCount = 1;
        ps[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps[2].descriptorCount = 5;
        VkDescriptorPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.maxSets = 1;
        pi.poolSizeCount = 3;
        pi.pPoolSizes = ps;
        VKC_CHECK(vkCreateDescriptorPool(g.device, &pi, nullptr, &g.pool));
    }
    {
        VkCommandPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.queueFamilyIndex = g.qfam;
        VKC_CHECK(vkCreateCommandPool(g.device, &ci, nullptr, &g.cmd_pool));
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = g.cmd_pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VKC_CHECK(vkAllocateCommandBuffers(g.device, &ai, &g.cmd));
        VkQueryPoolCreateInfo qi{};
        qi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = 2;
        VKC_CHECK(vkCreateQueryPool(g.device, &qi, nullptr, &g.query_pool));
    }
}

// Upload 6 blobs as UBO + 5 SSBOs, wire descriptor set.
// bufs[0] = camera UBO, [1..3] = prim SSBOs, [4] = BVH nodes,
// [5] = leaf refs (empty blobs get 16B pad: zero-size is illegal).
inline void gpu_set_scene(GpuContext &g, const void *data[6], const size_t bytes[6]) {
    for (int i = 0; i < 6; ++i) {
        size_t n = bytes[i] ? bytes[i] : 16;
        VkBufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size = n;
        ci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VKC_CHECK(vkCreateBuffer(g.device, &ci, nullptr, &g.scene_bufs[i].buf));
        g.scene_bufs[i].bytes = n;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(g.device, g.scene_bufs[i].buf, &req);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex =
            vkc_find_memory(g.phys, req.memoryTypeBits,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VKC_CHECK(vkAllocateMemory(g.device, &ai, nullptr, &g.scene_bufs[i].mem));
        VKC_CHECK(vkBindBufferMemory(g.device, g.scene_bufs[i].buf, g.scene_bufs[i].mem, 0));
        if (data[i]) {
            void *mapped = nullptr;
            VKC_CHECK(vkMapMemory(g.device, g.scene_bufs[i].mem, 0, bytes[i], 0, &mapped));
            std::memcpy(mapped, data[i], bytes[i]);
            vkUnmapMemory(g.device, g.scene_bufs[i].mem);
        }
    }
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = g.pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &g.layout;
    VKC_CHECK(vkAllocateDescriptorSets(g.device, &ai, &g.set));
    VkDescriptorImageInfo ii{};
    ii.imageView = g.view;
    ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorBufferInfo bi[6]{};
    for (int i = 0; i < 6; ++i) {
        bi[i].buffer = g.scene_bufs[i].buf;
        bi[i].range = g.scene_bufs[i].bytes;
    }
    VkWriteDescriptorSet w[7]{};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = g.set;
    w[0].dstBinding = 0;
    w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[0].pImageInfo = &ii;
    for (int i = 1; i < 7; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = g.set;
        w[i].dstBinding = (uint32_t)i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = (i == 1) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                       : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i - 1];
    }
    vkUpdateDescriptorSets(g.device, 7, w, 0, nullptr);
    g.has_scene = true;
}

// Dispatch spv with 32B push block, copy image to host, report device ms.
// out_rgba receives W*H*4 floats, top-first rows.
inline double gpu_run(GpuContext &g, const std::string &spv_path, const int push8[8],
                      std::vector<float> &out_rgba) {
    std::ifstream f(spv_path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "missing shader: " << spv_path << "\n";
        std::exit(1);
    }
    size_t n = (size_t)f.tellg();
    f.seekg(0);
    std::vector<char> code(n);
    f.read(code.data(), (std::streamsize)n);

    VkShaderModule mod;
    {
        VkShaderModuleCreateInfo mi{};
        mi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        mi.codeSize = code.size();
        mi.pCode = (const uint32_t *)code.data();
        VKC_CHECK(vkCreateShaderModule(g.device, &mi, nullptr, &mod));
    }
    {
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.size = 32;
        VkPipelineLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        li.setLayoutCount = 1;
        li.pSetLayouts = &g.layout;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &pc;
        VKC_CHECK(vkCreatePipelineLayout(g.device, &li, nullptr, &g.pipe_layout));
        VkComputePipelineCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pi.stage.module = mod;
        pi.stage.pName = "main";
        pi.layout = g.pipe_layout;
        VKC_CHECK(vkCreateComputePipelines(g.device, VK_NULL_HANDLE, 1, &pi, nullptr, &g.pipe));
        vkDestroyShaderModule(g.device, mod, nullptr);
    }
    {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VKC_CHECK(vkBeginCommandBuffer(g.cmd, &bi));
        vkCmdResetQueryPool(g.cmd, g.query_pool, 0, 2);
        VkImageMemoryBarrier b1{};
        b1.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b1.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b1.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b1.image = g.image;
        b1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b1.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &b1);
        vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipe);
        vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipe_layout, 0,
                                1, &g.set, 0, nullptr);
        vkCmdPushConstants(g.cmd, g.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, push8);
        vkCmdWriteTimestamp(g.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, g.query_pool, 0);
        vkCmdDispatch(g.cmd, (g.W + 15) / 16, (g.H + 15) / 16, 1);
        vkCmdWriteTimestamp(g.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, g.query_pool, 1);
        VkImageMemoryBarrier b2 = b1;
        b2.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b2.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b2.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                             1, &b2);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {(uint32_t)g.W, (uint32_t)g.H, 1};
        vkCmdCopyImageToBuffer(g.cmd, g.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               g.staging.buf, 1, &region);
        VKC_CHECK(vkEndCommandBuffer(g.cmd));
    }
    double dispatch_ms = 0;
    {
        VkFence fence;
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VKC_CHECK(vkCreateFence(g.device, &fi, nullptr, &fence));
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &g.cmd;
        VKC_CHECK(vkQueueSubmit(g.queue, 1, &si, fence));
        VKC_CHECK(vkWaitForFences(g.device, 1, &fence, VK_TRUE, UINT64_MAX));
        vkDestroyFence(g.device, fence, nullptr);
        uint64_t stamps[2] = {};
        VKC_CHECK(vkGetQueryPoolResults(g.device, g.query_pool, 0, 2, sizeof stamps,
                                        stamps, sizeof(uint64_t),
                                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(g.phys, &props);
        dispatch_ms = (double)(stamps[1] - stamps[0]) * props.limits.timestampPeriod / 1e6;
    }
    {
        void *mapped = nullptr;
        VKC_CHECK(vkMapMemory(g.device, g.staging.mem, 0, g.staging.bytes, 0, &mapped));
        out_rgba.assign((const float *)mapped,
                        (const float *)mapped + (size_t)g.W * g.H * 4);
        vkUnmapMemory(g.device, g.staging.mem);
    }
    vkDestroyPipeline(g.device, g.pipe, nullptr);
    g.pipe = VK_NULL_HANDLE;
    vkDestroyPipelineLayout(g.device, g.pipe_layout, nullptr);
    g.pipe_layout = VK_NULL_HANDLE;
    return dispatch_ms;
}

inline void gpu_shutdown(GpuContext &g) {
    if (g.pipe)
        vkDestroyPipeline(g.device, g.pipe, nullptr);
    if (g.pipe_layout)
        vkDestroyPipelineLayout(g.device, g.pipe_layout, nullptr);
    for (auto &hb : g.scene_bufs) {
        if (hb.buf)
            vkDestroyBuffer(g.device, hb.buf, nullptr);
        if (hb.mem)
            vkFreeMemory(g.device, hb.mem, nullptr);
    }
    vkDestroyBuffer(g.device, g.staging.buf, nullptr);
    vkFreeMemory(g.device, g.staging.mem, nullptr);
    vkDestroyImageView(g.device, g.view, nullptr);
    vkDestroyImage(g.device, g.image, nullptr);
    vkFreeMemory(g.device, g.image_mem, nullptr);
    vkDestroyQueryPool(g.device, g.query_pool, nullptr);
    vkDestroyDescriptorPool(g.device, g.pool, nullptr);
    vkDestroyDescriptorSetLayout(g.device, g.layout, nullptr);
    vkDestroyCommandPool(g.device, g.cmd_pool, nullptr);
    vkDestroyDevice(g.device, nullptr);
    vkDestroyInstance(g.instance, nullptr);
}
