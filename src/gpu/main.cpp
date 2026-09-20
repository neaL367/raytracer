// rt_gpu: headless Vulkan compute backend. CPU binary untouched.
// Usage: rt_gpu [shader.spv] [out.ppm] [--spp N] [--seed S]
// Shaders: grad (smoke), normal (silhouette), path (full transport).
#include "output/ppm.h"

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define VK_CHECK(x)                                                  \
    do {                                                             \
        VkResult r = (x);                                            \
        if (r != VK_SUCCESS) {                                       \
            std::cerr << "vulkan error " << r << " at " << __LINE__  \
                      << "\n";                                       \
            std::exit(1);                                            \
        }                                                            \
    } while (0)

namespace {
const int W = 400;
const int H = 225;

// std430 mirrors of common.glsl structs (vec4-packed, sequential).
struct GPUSphere {
    float c[4], alb[4], emit[4], prm[4];
};
struct GPUQuad {
    float Q[4], u[4], v[4], alb[4], emit[4], prm[4];
};
struct GPUTri {
    float a[4], b[4], c[4], alb[4], emit[4], prm[4];
};
struct GPUCam {
    float o[4], ll[4], h[4], v[4];
};

uint32_t find_memory(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(pd, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & want) == want)
            return i;
    std::cerr << "no suitable memory type\n";
    std::exit(1);
}

std::vector<char> read_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "missing shader: " << path << "\n";
        std::exit(1);
    }
    size_t n = (size_t)f.tellg();
    f.seekg(0);
    std::vector<char> data(n);
    f.read(data.data(), (std::streamsize)n);
    return data;
}

struct HostBuffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
};

HostBuffer make_upload_buffer(VkDevice dev, VkPhysicalDevice pd, size_t bytes,
                              const void *src) {
    HostBuffer hb;
    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = bytes ? bytes : 16; // Vulkan forbids zero-size buffers
    ci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(dev, &ci, nullptr, &hb.buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, hb.buf, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex =
        find_memory(pd, req.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(dev, &ai, nullptr, &hb.mem));
    VK_CHECK(vkBindBufferMemory(dev, hb.buf, hb.mem, 0));
    if (src) {
        void *mapped = nullptr;
        VK_CHECK(vkMapMemory(dev, hb.mem, 0, bytes, 0, &mapped));
        std::memcpy(mapped, src, bytes);
        vkUnmapMemory(dev, hb.mem);
    }
    return hb;
}
} // namespace

int main(int argc, char **argv) {
    std::string shader = SHADER_DIR "/grad.spv";
    std::string out_path = "out/gpu_grad.ppm";
    int spp = 16;
    int seed = 42;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--spp" && i + 1 < argc)
            spp = std::max(1, std::atoi(argv[++i]));
        else if (a == "--seed" && i + 1 < argc)
            seed = std::atoi(argv[++i]);
        else if (a.ends_with(".spv"))
            shader = a;
        else if (a.ends_with(".ppm"))
            out_path = a;
    }
    auto t0 = std::chrono::high_resolution_clock::now();

    VkInstance instance;
    {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "rt_gpu";
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &app;
        VK_CHECK(vkCreateInstance(&ci, nullptr, &instance));
    }

    // Prefer discrete NVIDIA (the 1650 Ti), else first compute-capable.
    VkPhysicalDevice chosen = VK_NULL_HANDLE;
    uint32_t chosen_qfam = 0;
    {
        uint32_t n = 0;
        VK_CHECK(vkEnumeratePhysicalDevices(instance, &n, nullptr));
        std::vector<VkPhysicalDevice> pds(n);
        VK_CHECK(vkEnumeratePhysicalDevices(instance, &n, pds.data()));
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
                chosen = pd;
                chosen_qfam = (uint32_t)qfam;
            }
        }
        if (chosen == VK_NULL_HANDLE) {
            std::cerr << "no compute device\n";
            return 1;
        }
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(chosen, &props);
        std::cout << "gpu: " << props.deviceName << "\n";
    }

    VkDevice device;
    VkQueue queue;
    {
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qi{};
        qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = chosen_qfam;
        qi.queueCount = 1;
        qi.pQueuePriorities = &prio;
        VkDeviceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.queueCreateInfoCount = 1;
        ci.pQueueCreateInfos = &qi;
        VK_CHECK(vkCreateDevice(chosen, &ci, nullptr, &device));
        vkGetDeviceQueue(device, chosen_qfam, 0, &queue);
    }

    // Scene mirrors CPU main: ground + 3 spheres + tri + ceiling light.
    // mat types: 0 lambertian, 1 metal, 2 dielectric, 3 emissive.
    // params = (type, fuzz, ir, 0). Leading quads emissive (nlights).
    const float aspect = (float)((double)W / (double)H);
    GPUCam gcam = {{0, 0, 0, 0},
                   {-aspect, -1, -1, 0},
                   {2 * aspect, 0, 0, 0},
                   {0, 2, 0, 0}};
    auto sph = [](float x, float y, float z, float r, float ar, float ag, float ab,
                  float er, float eg, float eb, float type, float fuzz, float ir) {
        GPUSphere s{};
        s.c[0] = x;
        s.c[1] = y;
        s.c[2] = z;
        s.c[3] = r;
        s.alb[0] = ar;
        s.alb[1] = ag;
        s.alb[2] = ab;
        s.emit[0] = er;
        s.emit[1] = eg;
        s.emit[2] = eb;
        s.prm[0] = type;
        s.prm[1] = fuzz;
        s.prm[2] = ir;
        return s;
    };
    std::vector<GPUSphere> spheres = {
        sph(0, -100.5f, -1, 100, 0.5f, 0.5f, 0.5f, 0, 0, 0, 0, 0, 0),
        sph(0, 0, -1, 0.5f, 0.7f, 0.3f, 0.3f, 0, 0, 0, 0, 0, 0),
        sph(-1, 0, -1, 0.5f, 0.8f, 0.8f, 0.8f, 0, 0, 0, 1, 0.3f, 0),
        sph(1, 0, -1, 0.5f, 0, 0, 0, 0, 0, 0, 2, 0, 1.5f),
    };
    GPUQuad light{};
    {
        float Q[4] = {-1, 1.9f, -2, 0}, u[4] = {2, 0, 0, 0}, v[4] = {0, 0, 2, 0};
        std::memcpy(light.Q, Q, sizeof Q);
        std::memcpy(light.u, u, sizeof u);
        std::memcpy(light.v, v, sizeof v);
        light.emit[0] = light.emit[1] = light.emit[2] = 4.0f;
        light.prm[0] = 3;
    }
    std::vector<GPUQuad> quads = {light};
    GPUTri tri{};
    {
        float a[4] = {-0.3f, -0.35f, -0.6f, 0}, b[4] = {0.3f, -0.35f, -0.6f, 0},
              c[4] = {0, 0.1f, -0.6f, 0}, alb[4] = {0.7f, 0.3f, 0.3f, 0};
        std::memcpy(tri.a, a, sizeof a);
        std::memcpy(tri.b, b, sizeof b);
        std::memcpy(tri.c, c, sizeof c);
        std::memcpy(tri.alb, alb, sizeof alb);
    }
    std::vector<GPUTri> tris = {tri};

    HostBuffer cam_buf = make_upload_buffer(device, chosen, sizeof gcam, &gcam);
    HostBuffer sph_buf =
        make_upload_buffer(device, chosen, spheres.size() * sizeof(GPUSphere), spheres.data());
    HostBuffer quad_buf =
        make_upload_buffer(device, chosen, quads.size() * sizeof(GPUQuad), quads.data());
    HostBuffer tri_buf =
        make_upload_buffer(device, chosen, tris.size() * sizeof(GPUTri), tris.data());

    // Storage image RGBA32F + host staging buffer.
    VkImage image;
    VkDeviceMemory image_mem;
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
        VK_CHECK(vkCreateImage(device, &ci, nullptr, &image));
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device, image, &req);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = find_memory(chosen, req.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &image_mem));
        VK_CHECK(vkBindImageMemory(device, image, image_mem, 0));
    }
    VkImageView view;
    {
        VkImageViewCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image = image;
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(device, &ci, nullptr, &view));
    }
    const size_t stage_size = (size_t)W * H * 16;
    VkBuffer staging;
    VkDeviceMemory staging_mem;
    {
        VkBufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size = stage_size;
        ci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(device, &ci, nullptr, &staging));
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, staging, &req);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex =
            find_memory(chosen, req.memoryTypeBits,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &staging_mem));
        VK_CHECK(vkBindBufferMemory(device, staging, staging_mem, 0));
    }

    // Descriptors: 0 image, 1 camera UBO, 2-4 scene SSBOs.
    VkDescriptorSetLayout layout;
    VkDescriptorPool pool;
    VkDescriptorSet set;
    {
        VkDescriptorSetLayoutBinding b[5]{};
        b[0].binding = 0;
        b[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[0].descriptorCount = 1;
        b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        for (uint32_t i = 1; i < 5; ++i) {
            b[i].binding = i;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            b[i].descriptorType = (i == 1) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                           : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 5;
        ci.pBindings = b;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &ci, nullptr, &layout));
        VkDescriptorPoolSize ps[3]{};
        ps[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        ps[0].descriptorCount = 1;
        ps[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ps[1].descriptorCount = 1;
        ps[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps[2].descriptorCount = 3;
        VkDescriptorPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.maxSets = 1;
        pi.poolSizeCount = 3;
        pi.pPoolSizes = ps;
        VK_CHECK(vkCreateDescriptorPool(device, &pi, nullptr, &pool));
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &layout;
        VK_CHECK(vkAllocateDescriptorSets(device, &ai, &set));
        VkDescriptorImageInfo ii{};
        ii.imageView = view;
        ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorBufferInfo bi[4]{};
        bi[0].buffer = cam_buf.buf;
        bi[0].range = sizeof gcam;
        bi[1].buffer = sph_buf.buf;
        bi[1].range = spheres.size() * sizeof(GPUSphere);
        bi[2].buffer = quad_buf.buf;
        bi[2].range = quads.size() * sizeof(GPUQuad);
        bi[3].buffer = tri_buf.buf;
        bi[3].range = tris.size() * sizeof(GPUTri);
        VkWriteDescriptorSet w[5]{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = set;
        w[0].dstBinding = 0;
        w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[0].pImageInfo = &ii;
        for (int i = 1; i < 5; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = set;
            w[i].dstBinding = (uint32_t)i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = (i == 1) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                           : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &bi[i - 1];
        }
        vkUpdateDescriptorSets(device, 5, w, 0, nullptr);
    }

    // Pipeline + 32B push constants (W,H,ns,nq,nt,spp,seed,nlights).
    VkPipeline pipe;
    VkPipelineLayout pipe_layout;
    {
        auto code = read_file(shader);
        VkShaderModuleCreateInfo mi{};
        mi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        mi.codeSize = code.size();
        mi.pCode = (const uint32_t *)code.data();
        VkShaderModule mod;
        VK_CHECK(vkCreateShaderModule(device, &mi, nullptr, &mod));
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.size = 32;
        VkPipelineLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        li.setLayoutCount = 1;
        li.pSetLayouts = &layout;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(device, &li, nullptr, &pipe_layout));
        VkComputePipelineCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pi.stage.module = mod;
        pi.stage.pName = "main";
        pi.layout = pipe_layout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pi, nullptr, &pipe));
        vkDestroyShaderModule(device, mod, nullptr);
    }

    VkCommandPool cmd_pool;
    VkCommandBuffer cmd;
    VkQueryPool query_pool;
    {
        VkQueryPoolCreateInfo qi{};
        qi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = 2;
        VK_CHECK(vkCreateQueryPool(device, &qi, nullptr, &query_pool));
    }
    {
        VkCommandPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.queueFamilyIndex = chosen_qfam;
        VK_CHECK(vkCreateCommandPool(device, &ci, nullptr, &cmd_pool));
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = cmd_pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(device, &ai, &cmd));
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        vkCmdResetQueryPool(cmd, query_pool, 0, 2);
        VkImageMemoryBarrier b1{};
        b1.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b1.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b1.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b1.image = image;
        b1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b1.srcAccessMask = 0;
        b1.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &b1);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0, 1,
                                &set, 0, nullptr);
        int pc[8] = {W, H, (int)spheres.size(), (int)quads.size(), (int)tris.size(),
                     spp, seed, 1};
        vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, pc);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, 0);
        vkCmdDispatch(cmd, (W + 15) / 16, (H + 15) / 16, 1);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, 1);
        VkImageMemoryBarrier b2 = b1;
        b2.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b2.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b2.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                             1, &b2);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {(uint32_t)W, (uint32_t)H, 1};
        vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               staging, 1, &region);
        VK_CHECK(vkEndCommandBuffer(cmd));
    }
    {
        VkFence fence;
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VK_CHECK(vkCreateFence(device, &fi, nullptr, &fence));
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
        vkDestroyFence(device, fence, nullptr);
        {
            uint64_t stamps[2] = {};
            VK_CHECK(vkGetQueryPoolResults(device, query_pool, 0, 2, sizeof stamps,
                                           stamps, sizeof(uint64_t),
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(chosen, &props);
            double ms = (double)(stamps[1] - stamps[0]) * props.limits.timestampPeriod / 1e6;
            std::cout << "device dispatch: " << ms << "ms\n";
        }
    }

    {
        void *mapped = nullptr;
        VK_CHECK(vkMapMemory(device, staging_mem, 0, stage_size, 0, &mapped));
        const float *px = (const float *)mapped;
        std::vector<vec3> fb((size_t)W * H);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const float *t = px + ((size_t)y * W + x) * 4;
                fb[((size_t)H - 1 - y) * W + x] = vec3(t[0], t[1], t[2]);
            }
        vkUnmapMemory(device, staging_mem);
        std::filesystem::create_directories("out");
        if (!write_ppm(out_path.c_str(), fb, W, H)) {
            std::cerr << "write failed\n";
            return 1;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "wrote " << out_path << " spp=" << spp << " in "
              << std::chrono::duration<double>(t1 - t0).count() << "s\n";

    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, staging_mem, nullptr);
    for (auto hb : {cam_buf, sph_buf, quad_buf, tri_buf}) {
        vkDestroyBuffer(device, hb.buf, nullptr);
        vkFreeMemory(device, hb.mem, nullptr);
    }
    vkDestroyImageView(device, view, nullptr);
    vkDestroyImage(device, image, nullptr);
    vkFreeMemory(device, image_mem, nullptr);
    vkDestroyPipeline(device, pipe, nullptr);
    vkDestroyPipelineLayout(device, pipe_layout, nullptr);
    vkDestroyQueryPool(device, query_pool, nullptr);
    vkDestroyDescriptorPool(device, pool, nullptr);
    vkDestroyDescriptorSetLayout(device, layout, nullptr);
    vkDestroyCommandPool(device, cmd_pool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
}
