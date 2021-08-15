extern crate ash;
extern crate sdl2;

use crate::math::fft::Complex;
use crate::math::lin_alg::{Mat4, Vec2, Vec3, Vec4};
use crate::math::rand::{box_muller_rng, xorshift32};
use crate::vk_initializers;

use ash::extensions::{
    ext::DebugUtils,
    khr::{Surface, Swapchain},
    nv::MeshShader,
};
use ash::version::{DeviceV1_0, InstanceV1_0, InstanceV1_1};
use ash::{vk, Device, Instance};
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::video::Window;
use std::collections::HashMap;
use std::mem::size_of;

struct VkBuffer {
    buffer: vk::Buffer,
    buffer_memory: vk::DeviceMemory,
}

struct VkImage {
    image: vk::Image,
    image_memory: vk::DeviceMemory,
}

#[repr(align(16))]
struct Vertex {
    pos: Vec3,
    norm: Vec3,
}

#[derive(Debug, Clone, Copy)]
struct Meshlet {
    vertices: [u32; 64],
    primitives: [u32; 42],
    vertex_and_index_count: u16,
}

struct Mesh {
    vertices: Vec<Vertex>,
    vertex_buffer: VkBuffer,
    indices: Vec<u32>,
    index_buffer: VkBuffer,
}

struct MeshShaderData {
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    buffers: [VkBuffer; 4],
    meshlet_count: u32,
}

struct Material {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

struct RenderObject {
    mesh_key: String,
    material_key: String,
    transformation_matrix: Mat4,
}

struct SceneData {
    view: Mat4,
    projection: Mat4,
    fog_col: Vec4,
    fog_distances: Vec4,
    ambient_color: Vec4,
    sun_light_dir: Vec4,
    sun_light_col: Vec4,
}

const FRAME_OVERLAP: usize = 2;
const MAX_OBJECTS: usize = 10_000;
const OCEAN_PATCH_DIM: usize = 512;
const L_X: f32 = 1000.0f32;
const L_Y: f32 = 1000.0f32;
const TWO_PI: f32 = std::f32::consts::PI * 2.0;

struct FrameData {
    present_semaphore: vk::Semaphore,
    render_semaphore: vk::Semaphore,
    render_fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
    object_buffer: VkBuffer,
    object_descriptor: vk::DescriptorSet,
}

impl Vertex {
    pub fn construct_vertices_from_positions(
        positions: Vec<Vec3>,
        normals: Vec<Vec3>,
    ) -> Vec<Vertex> {
        assert!(positions.len() == normals.len());

        let mut result = Vec::<Vertex>::with_capacity(positions.len());
        for (i, _) in positions.iter().enumerate() {
            result.push(Vertex {
                pos: positions[i],
                norm: normals[i],
            });
        }
        return result;
    }
}

struct Camera {
    pos: Vec3,
    front: Vec3,
    up: Vec3,
    yaw: f32,
    pitch: f32,
    fov: f32,
}

impl Default for Camera {
    fn default() -> Self {
        #[rustfmt::skip]
        return Camera {
            pos: Vec3 { x: 0.0, y: 0.0, z: 3.0 },
            front: Vec3 { x: 0.0, y: 0.0, z: -1.0 },
            up: Vec3::up(),
            yaw: -90.0,
            pitch: 0.0,
            fov: 45.0,
        };
    }
}

struct VkPipelineBuilder {
    shader_stages: Vec<vk::PipelineShaderStageCreateInfo>,
    vertex_input_info: vk::PipelineVertexInputStateCreateInfo,
    input_assembly: vk::PipelineInputAssemblyStateCreateInfo,
    viewport: vk::Viewport,
    scissor: vk::Rect2D,
    rasterizer: vk::PipelineRasterizationStateCreateInfo,
    color_blend_attachment: vk::PipelineColorBlendAttachmentState,
    multisampling: vk::PipelineMultisampleStateCreateInfo,
    depth_stencil_state: vk::PipelineDepthStencilStateCreateInfo,
    pipeline_layout: vk::PipelineLayout,
}

impl VkPipelineBuilder {
    pub fn default() -> VkPipelineBuilder {
        return VkPipelineBuilder {
            shader_stages: Vec::<vk::PipelineShaderStageCreateInfo>::default(),
            vertex_input_info: vk::PipelineVertexInputStateCreateInfo::default(),
            input_assembly: vk::PipelineInputAssemblyStateCreateInfo::default(),
            viewport: vk::Viewport::default(),
            scissor: vk::Rect2D::default(),
            rasterizer: vk::PipelineRasterizationStateCreateInfo::default(),
            color_blend_attachment: vk::PipelineColorBlendAttachmentState::default(),
            multisampling: vk::PipelineMultisampleStateCreateInfo::default(),
            depth_stencil_state: vk::PipelineDepthStencilStateCreateInfo::default(),
            pipeline_layout: vk::PipelineLayout::default(),
        };
    }

    pub fn build_pipline(&self, render_pass: &vk::RenderPass, device: &Device) -> vk::Pipeline {
        let viewports = [self.viewport];
        let scissors = [self.scissor];
        let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        let attachments = [self.color_blend_attachment];
        let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&attachments);

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(self.shader_stages.as_slice())
            .vertex_input_state(&self.vertex_input_info)
            .input_assembly_state(&self.input_assembly)
            .viewport_state(&viewport_state_create_info)
            .rasterization_state(&self.rasterizer)
            .multisample_state(&self.multisampling)
            .color_blend_state(&color_blend_state_create_info)
            .depth_stencil_state(&self.depth_stencil_state)
            .layout(self.pipeline_layout)
            .render_pass(*render_pass)
            .build();
        let pipeline_create_infos = [pipeline_create_info];
        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_create_infos, None)
                .unwrap()
        };

        return pipeline[0];
    }
}

pub struct VkEngine {
    sdl_context: sdl2::Sdl,
    window: Window,
    size: vk::Extent2D,
    frame_count: u32,
    instance: Instance,
    debug_utils_loader: DebugUtils,
    debug_callback: vk::DebugUtilsMessengerEXT,
    surface: vk::SurfaceKHR,
    surface_loader: Surface,
    physical_device_properties: vk::PhysicalDeviceProperties,
    device: Device,
    swapchain_loader: Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_image_views: Vec<vk::ImageView>,
    depth_image: VkImage,
    depth_image_view: vk::ImageView,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    graphics_queue: vk::Queue,
    global_set_layout: vk::DescriptorSetLayout,
    global_descriptor_set: vk::DescriptorSet,
    object_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    command_pool: vk::CommandPool,
    frame_data: [FrameData; FRAME_OVERLAP],
    scene_data_buffer: VkBuffer,
    materials: HashMap<String, Material>,
    camera: Camera,
    last_timestamp: std::time::Instant,
    mesh_shader_data: MeshShaderData,
    mesh_shader: MeshShader,
    compute_pipelines: Vec<vk::Pipeline>,
    start: std::time::Instant,
}

impl VkEngine {
    pub fn new(width: u32, height: u32) -> VkEngine {
        let (sdl_context, window) = vk_initializers::create_sdl_window(width, height);
        sdl_context.mouse().set_relative_mouse_mode(true);

        let (entry, instance) = vk_initializers::create_instance(&window);

        let (debug_utils_loader, debug_callback) =
            vk_initializers::create_debug_layer(&entry, &instance);

        let (surface, surface_loader) = vk_initializers::create_surface(&window, &entry, &instance);

        let (physical_device, queue_family_index) =
            vk_initializers::get_physical_device_and_graphics_queue_family_index(
                &instance,
                &surface_loader,
                &surface,
            );

        let physical_device_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };

        let device =
            vk_initializers::create_device(queue_family_index, &instance, &physical_device);

        let surface_format = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()[0]
        };

        let (swapchain_loader, swapchain) = vk_initializers::create_swapchain_loader_and_swapchain(
            &surface_loader,
            &physical_device,
            &surface,
            width,
            height,
            &instance,
            &device,
            &surface_format,
        );

        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain).unwrap() };
        let swapchain_image_views = vk_initializers::create_swapchain_image_views(
            &swapchain_images,
            &surface_format,
            &device,
        );

        let depth_image_extent = vk::Extent3D::builder()
            .width(width)
            .height(height)
            .depth(1)
            .build();
        let depth_image_format = vk::Format::D32_SFLOAT;

        let depth_image_create_info = vk_initializers::create_image_create_info(
            depth_image_format,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            depth_image_extent,
        );
        let (depth_image, depth_image_memory) = vk_initializers::create_image(
            &instance,
            physical_device,
            &device,
            depth_image_create_info,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let depth_image = VkImage {
            image: depth_image,
            image_memory: depth_image_memory,
        };

        let depth_image_view_create_info = vk_initializers::create_imageview_create_info(
            depth_image_format,
            depth_image.image,
            vk::ImageAspectFlags::DEPTH,
        );
        let depth_image_view = unsafe {
            device
                .create_image_view(&depth_image_view_create_info, None)
                .unwrap()
        };

        let render_pass = vk_initializers::create_renderpass(&surface_format, &device);

        let framebuffers = vk_initializers::create_framebuffers(
            &swapchain_image_views,
            depth_image_view,
            &render_pass,
            width,
            height,
            &device,
        );

        let graphics_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let scene_buffer_binding = vk_initializers::descriptor_set_layout_binding(
            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            0,
            1,
            vk::ShaderStageFlags::COMPUTE
                | vk::ShaderStageFlags::MESH_NV
                | vk::ShaderStageFlags::FRAGMENT,
        );

        let bindings = [scene_buffer_binding];
        let global_set_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        let global_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&global_set_info, None)
                .unwrap()
        };

        let object_buffer_binding = vk_initializers::descriptor_set_layout_binding(
            vk::DescriptorType::STORAGE_BUFFER,
            0,
            1,
            vk::ShaderStageFlags::VERTEX,
        );

        let bindings = [object_buffer_binding];
        let object_set_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        let object_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&object_set_info, None)
                .unwrap()
        };

        let pool_sizes = [
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(10)
                .build(),
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(10)
                .build(),
        ];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(10)
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .unwrap()
        };

        let set_layouts = [global_set_layout];
        let global_descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let global_descriptor_set = unsafe {
            device
                .allocate_descriptor_sets(&global_descriptor_set_allocate_info)
                .unwrap()[0]
        };

        let scene_param_buffer_size = FRAME_OVERLAP
            * Self::return_aligned_size(physical_device_properties, size_of::<SceneData>());
        let (scene_param_buffer, scene_param_buffer_memory) = vk_initializers::create_buffer(
            &instance,
            physical_device,
            &device,
            scene_param_buffer_size as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let scene_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(scene_param_buffer)
            .offset(0)
            .range(size_of::<SceneData>() as u64)
            .build();
        let scene_buffer_infos = [scene_buffer_info];
        let scene_set_write = vk::WriteDescriptorSet::builder()
            .dst_binding(0)
            .dst_set(global_descriptor_set)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .buffer_info(&scene_buffer_infos)
            .build();

        unsafe { device.update_descriptor_sets(&[scene_set_write], &[]) };

        let (command_pool, command_buffers) = vk_initializers::create_command_pool_and_buffer(
            queue_family_index,
            &device,
            FRAME_OVERLAP as u32,
        );
        let mut frame_data: [FrameData; FRAME_OVERLAP] = unsafe { std::mem::zeroed() };
        for i in 0..FRAME_OVERLAP {
            let semaphore_create_info = vk::SemaphoreCreateInfo::default();
            let render_semaphore = unsafe {
                device
                    .create_semaphore(&semaphore_create_info, None)
                    .unwrap()
            };
            let present_semaphore = unsafe {
                device
                    .create_semaphore(&semaphore_create_info, None)
                    .unwrap()
            };

            let fence_create_info =
                vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
            let render_fence = unsafe { device.create_fence(&fence_create_info, None).unwrap() };

            frame_data[i] = FrameData {
                present_semaphore,
                render_semaphore,
                render_fence,
                command_buffer: command_buffers[i],
                object_buffer: VkBuffer {
                    buffer: unsafe { std::mem::zeroed() },
                    buffer_memory: unsafe { std::mem::zeroed() },
                },
                object_descriptor: unsafe { std::mem::zeroed() },
            };
        }

        let mut tilde_h_zero: Vec<Complex> =
            vec![unsafe { std::mem::zeroed() }; OCEAN_PATCH_DIM * OCEAN_PATCH_DIM];
        let mut tilde_h_conjugate_zero: Vec<Complex> =
            vec![unsafe { std::mem::zeroed() }; OCEAN_PATCH_DIM * OCEAN_PATCH_DIM];

        let amplitude = 20.0;
        let wind_speed = 31.0;
        let wind_direction = Vec2 { x: 1.0, y: 0.0 };
        let g = 9.81;
        let l = wind_speed * wind_speed / g;

        let mut rnd_state = 1;

        for i in 0..OCEAN_PATCH_DIM {
            for j in 0..OCEAN_PATCH_DIM {
                let k = Vec2 {
                    x: (i as f32 - (OCEAN_PATCH_DIM as f32 / 2.0)) * TWO_PI / L_X,
                    y: (j as f32 - (OCEAN_PATCH_DIM as f32 / 2.0)) * TWO_PI / L_Y,
                };
                let k_length_sqr = if k.length_sqr() < (0.0001 * 0.0001) {
                    0.0001 * 0.0001
                } else {
                    k.length_sqr()
                };

                let b = f32::exp(-1.0 / (k_length_sqr * l * l)) / (k_length_sqr * k_length_sqr);
                let c = f32::powi(Vec2::dot(k.normal(), wind_direction.normal()), 2);

                let phillips_k = amplitude * b * c;

                let k = Vec2 {
                    x: -(i as f32 - (OCEAN_PATCH_DIM as f32 / 2.0)) * TWO_PI / L_X,
                    y: -(j as f32 - (OCEAN_PATCH_DIM as f32 / 2.0)) * TWO_PI / L_Y,
                };

                let c = f32::powi(Vec2::dot(k.normal(), wind_direction), 2);

                let phillips_minus_k = amplitude * b * c;

                let h_zero_k = f32::sqrt(phillips_k) / std::f32::consts::SQRT_2;
                let h_zero_minus_k = f32::sqrt(phillips_minus_k) / std::f32::consts::SQRT_2;

                let (u1, u2) = (xorshift32(&mut rnd_state), xorshift32(&mut rnd_state));
                let (z1, z2) =
                    box_muller_rng(u1 as f32 / u32::MAX as f32, u2 as f32 / u32::MAX as f32);

                let temp = Complex { real: z1, imag: z2 } * h_zero_k;
                tilde_h_zero[i * OCEAN_PATCH_DIM + j] = temp;

                let (u1, u2) = (xorshift32(&mut rnd_state), xorshift32(&mut rnd_state));
                let (z1, z2) =
                    box_muller_rng(u1 as f32 / u32::MAX as f32, u2 as f32 / u32::MAX as f32);

                tilde_h_conjugate_zero[i * OCEAN_PATCH_DIM + j] = Complex {
                    real: z1,
                    imag: -z2,
                } * h_zero_minus_k;
            }
        }

        let tilda_h_binding = vk_initializers::descriptor_set_layout_binding(
            vk::DescriptorType::STORAGE_BUFFER,
            0,
            1,
            vk::ShaderStageFlags::COMPUTE,
        );
        let tilda_h_conjugate_binding = vk_initializers::descriptor_set_layout_binding(
            vk::DescriptorType::STORAGE_BUFFER,
            1,
            1,
            vk::ShaderStageFlags::COMPUTE,
        );
        let ifft_output_input_binding = vk_initializers::descriptor_set_layout_binding(
            vk::DescriptorType::STORAGE_BUFFER,
            2,
            1,
            vk::ShaderStageFlags::COMPUTE,
        );
        let ifft_input_output_binding = vk_initializers::descriptor_set_layout_binding(
            vk::DescriptorType::STORAGE_BUFFER,
            3,
            1,
            vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::MESH_NV,
        );

        let bindings = [
            tilda_h_binding,
            tilda_h_conjugate_binding,
            ifft_output_input_binding,
            ifft_input_output_binding,
        ];
        let tilda_hs_descriptor_layout_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        let tilda_hs_descriptor_layout = unsafe {
            device
                .create_descriptor_set_layout(&tilda_hs_descriptor_layout_info, None)
                .unwrap()
        };

        let set_layouts = [tilda_hs_descriptor_layout];
        let tilda_hs_descriptor_allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let tilda_hs_descriptor_set = unsafe {
            device
                .allocate_descriptor_sets(&tilda_hs_descriptor_allocate_info)
                .unwrap()[0]
        };

        let tilda_h_size = (size_of::<Complex>() * tilde_h_zero.len()) as u64;
        let (temp_buffer, temp_buffer_memory) = vk_initializers::create_buffer(
            &instance,
            physical_device,
            &device,
            tilda_h_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data_ptr = device
                .map_memory(
                    temp_buffer_memory,
                    0,
                    tilda_h_size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap() as *mut Complex;
            data_ptr.copy_from_nonoverlapping(tilde_h_zero.as_ptr(), tilde_h_zero.len());
            device.unmap_memory(temp_buffer_memory);
        }

        let (tilda_h_buffer, tilda_h_buffer_memory) = vk_initializers::create_buffer(
            &instance,
            physical_device,
            &device,
            tilda_h_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        vk_initializers::copy_buffer(
            &device,
            command_pool,
            graphics_queue,
            temp_buffer,
            tilda_h_buffer,
            tilda_h_size,
        );

        unsafe {
            vk_initializers::free_buffer_and_memory(&device, temp_buffer, temp_buffer_memory)
        };

        let tilda_h_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(tilda_h_buffer)
            .offset(0)
            .range(tilda_h_size)
            .build();
        let tilda_h_buffer_infos = [tilda_h_buffer_info];
        let tilda_h_set_write = vk::WriteDescriptorSet::builder()
            .dst_set(tilda_hs_descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&tilda_h_buffer_infos)
            .build();

        let tilda_h_conjugate_size = (size_of::<Complex>() * tilde_h_conjugate_zero.len()) as u64;
        let (temp_buffer, temp_buffer_memory) = vk_initializers::create_buffer(
            &instance,
            physical_device,
            &device,
            tilda_h_conjugate_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data_ptr = device
                .map_memory(
                    temp_buffer_memory,
                    0,
                    tilda_h_conjugate_size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap() as *mut Complex;
            data_ptr.copy_from_nonoverlapping(
                tilde_h_conjugate_zero.as_ptr(),
                tilde_h_conjugate_zero.len(),
            );
            device.unmap_memory(temp_buffer_memory);
        }

        let (tilda_h_conjugate_buffer, tilda_h_conjugate_buffer_memory) =
            vk_initializers::create_buffer(
                &instance,
                physical_device,
                &device,
                tilda_h_conjugate_size,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );

        vk_initializers::copy_buffer(
            &device,
            command_pool,
            graphics_queue,
            temp_buffer,
            tilda_h_conjugate_buffer,
            tilda_h_conjugate_size,
        );

        unsafe {
            vk_initializers::free_buffer_and_memory(&device, temp_buffer, temp_buffer_memory)
        };

        let tilda_h_conjugate_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(tilda_h_conjugate_buffer)
            .offset(0)
            .range(tilda_h_conjugate_size)
            .build();
        let tilda_h_conjugate_buffer_infos = [tilda_h_conjugate_buffer_info];
        let tilda_h_conjugate_set_write = vk::WriteDescriptorSet::builder()
            .dst_set(tilda_hs_descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&tilda_h_conjugate_buffer_infos)
            .build();

        let ifft_output_input_size = (size_of::<Complex>() * tilde_h_zero.len()) as u64;
        let (ifft_output_input_buffer, ifft_output_input_buffer_memory) =
            vk_initializers::create_buffer(
                &instance,
                physical_device,
                &device,
                ifft_output_input_size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );

        let ifft_output_input_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(ifft_output_input_buffer)
            .offset(0)
            .range(ifft_output_input_size)
            .build();
        let ifft_output_input_buffer_infos = [ifft_output_input_buffer_info];
        let ifft_output_input_set_write = vk::WriteDescriptorSet::builder()
            .dst_set(tilda_hs_descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&ifft_output_input_buffer_infos)
            .build();

        let ifft_input_output_size = (size_of::<Complex>() * tilde_h_zero.len()) as u64;
        let (ifft_input_output_buffer, ifft_input_output_buffer_memory) =
            vk_initializers::create_buffer(
                &instance,
                physical_device,
                &device,
                ifft_input_output_size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );

        let ifft_input_output_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(ifft_input_output_buffer)
            .offset(0)
            .range(ifft_input_output_size)
            .build();
        let ifft_input_output_buffer_infos = [ifft_input_output_buffer_info];
        let ifft_input_output_set_write = vk::WriteDescriptorSet::builder()
            .dst_set(tilda_hs_descriptor_set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&ifft_input_output_buffer_infos)
            .build();

        unsafe {
            device.update_descriptor_sets(
                &[
                    tilda_h_set_write,
                    tilda_h_conjugate_set_write,
                    ifft_output_input_set_write,
                    ifft_input_output_set_write,
                ],
                &[],
            )
        };

        let push_constant_ranges = [vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(size_of::<Vec4>() as u32)
            .build()];

        let ocean_pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder()
                        .set_layouts(&[global_set_layout, tilda_hs_descriptor_layout])
                        .push_constant_ranges(&push_constant_ranges),
                    None,
                )
                .unwrap()
        };

        let ocean_mesh_shader_module =
            vk_initializers::create_shader_module(&device, "./shaders/ocean.mesh.spv");
        let ocean_fragment_shader_module =
            vk_initializers::create_shader_module(&device, "./shaders/ocean.frag.spv");

        let mut pipeline_builder = VkPipelineBuilder::default();

        pipeline_builder
            .shader_stages
            .push(vk_initializers::pipeline_shader_stage_create_info(
                vk::ShaderStageFlags::MESH_NV,
                ocean_mesh_shader_module,
                "ms_main\0",
            ));
        pipeline_builder
            .shader_stages
            .push(vk_initializers::pipeline_shader_stage_create_info(
                vk::ShaderStageFlags::FRAGMENT,
                ocean_fragment_shader_module,
                "fs_main\0",
            ));

        pipeline_builder.vertex_input_info = vk_initializers::vertex_input_state_create_info();

        pipeline_builder.input_assembly =
            vk_initializers::input_assembly_create_info(&vk::PrimitiveTopology::TRIANGLE_LIST);

        let (width_f32, height_f32) = (width as f32, height as f32);
        pipeline_builder.viewport = vk::Viewport::builder()
            .x(0.0f32)
            .y(0.0f32)
            .width(width_f32)
            .height(height_f32)
            .min_depth(0.0f32)
            .max_depth(1.0f32)
            .build();

        pipeline_builder.scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(vk::Extent2D { width, height })
            .build();

        pipeline_builder.rasterizer =
            vk_initializers::rasterization_state_create_info(vk::PolygonMode::LINE);

        pipeline_builder.multisampling = vk_initializers::multisampling_state_create_info();

        pipeline_builder.color_blend_attachment = vk_initializers::color_blend_attachment_state();

        pipeline_builder.depth_stencil_state = vk_initializers::create_depth_stencil_create_info();

        pipeline_builder.pipeline_layout = ocean_pipeline_layout;

        let ocean_pipeline = pipeline_builder.build_pipline(&render_pass, &device);

        unsafe {
            device.destroy_shader_module(ocean_mesh_shader_module, None);
            device.destroy_shader_module(ocean_fragment_shader_module, None);
        };

        let skybox_pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder().set_layouts(&[global_set_layout]),
                    None,
                )
                .unwrap()
        };

        let skybox_mesh_shader_module =
            vk_initializers::create_shader_module(&device, "./shaders/skybox.mesh.spv");
        let skybox_fragment_shader_module =
            vk_initializers::create_shader_module(&device, "./shaders/skybox.frag.spv");

        pipeline_builder.shader_stages.clear();
        pipeline_builder
            .shader_stages
            .push(vk_initializers::pipeline_shader_stage_create_info(
                vk::ShaderStageFlags::MESH_NV,
                skybox_mesh_shader_module,
                "ms_main\0",
            ));
        pipeline_builder
            .shader_stages
            .push(vk_initializers::pipeline_shader_stage_create_info(
                vk::ShaderStageFlags::FRAGMENT,
                skybox_fragment_shader_module,
                "fs_main\0",
            ));

        pipeline_builder.rasterizer =
            vk_initializers::rasterization_state_create_info(vk::PolygonMode::FILL);

        pipeline_builder.depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default();

        pipeline_builder.pipeline_layout = skybox_pipeline_layout;

        let skybox_pipeline = pipeline_builder.build_pipline(&render_pass, &device);

        unsafe {
            device.destroy_shader_module(skybox_mesh_shader_module, None);
            device.destroy_shader_module(skybox_fragment_shader_module, None);
        };

        let spectrum_and_ifft_shader_module =
            vk_initializers::create_shader_module(&device, "./shaders/ocean.comp.spv");

        let spectrum_and_ifft_pipline_create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(vk_initializers::pipeline_shader_stage_create_info(
                vk::ShaderStageFlags::COMPUTE,
                spectrum_and_ifft_shader_module,
                "cs_main\0",
            ))
            .layout(ocean_pipeline_layout)
            .build();
        let compute_infos = [spectrum_and_ifft_pipline_create_info];
        let compute_pipelines = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &compute_infos, None)
                .unwrap()
        };

        unsafe {
            device.destroy_shader_module(spectrum_and_ifft_shader_module, None);
        };

        let camera = Camera::default();

        let mut material_map = HashMap::<String, Material>::new();
        material_map.insert(
            "ocean".to_string(),
            Material {
                pipeline: ocean_pipeline,
                pipeline_layout: ocean_pipeline_layout,
            },
        );
        material_map.insert(
            "skybox".to_string(),
            Material {
                pipeline: skybox_pipeline,
                pipeline_layout: skybox_pipeline_layout,
            },
        );

        let mesh_shader = MeshShader::new(&instance, &device);

        return VkEngine {
            sdl_context,
            window,
            size: vk::Extent2D { width, height },
            frame_count: 0,
            instance,
            debug_utils_loader,
            debug_callback,
            surface,
            surface_loader,
            physical_device_properties,
            device,
            swapchain_loader,
            swapchain,
            swapchain_image_views,
            depth_image,
            depth_image_view,
            render_pass,
            framebuffers,
            graphics_queue,
            global_set_layout,
            global_descriptor_set,
            object_set_layout,
            descriptor_pool,
            command_pool,
            frame_data,
            scene_data_buffer: VkBuffer {
                buffer: scene_param_buffer,
                buffer_memory: scene_param_buffer_memory,
            },
            materials: material_map,
            camera,
            last_timestamp: std::time::Instant::now(),
            mesh_shader_data: MeshShaderData {
                descriptor_set_layout: tilda_hs_descriptor_layout,
                descriptor_set: tilda_hs_descriptor_set,
                buffers: [
                    VkBuffer {
                        buffer: tilda_h_buffer,
                        buffer_memory: tilda_h_buffer_memory,
                    },
                    VkBuffer {
                        buffer: tilda_h_conjugate_buffer,
                        buffer_memory: tilda_h_conjugate_buffer_memory,
                    },
                    VkBuffer {
                        buffer: ifft_output_input_buffer,
                        buffer_memory: ifft_output_input_buffer_memory,
                    },
                    VkBuffer {
                        buffer: ifft_input_output_buffer,
                        buffer_memory: ifft_input_output_buffer_memory,
                    },
                ],
                meshlet_count: 1,
            },
            mesh_shader,
            compute_pipelines,
            start: std::time::Instant::now(),
        };
    }

    fn return_aligned_size(
        physical_device_properties: vk::PhysicalDeviceProperties,
        original_size: usize,
    ) -> usize {
        let min_ubo_alignment = physical_device_properties
            .limits
            .min_uniform_buffer_offset_alignment as usize;
        let aligned_size = if min_ubo_alignment > 0 {
            (original_size + min_ubo_alignment - 1) & !(min_ubo_alignment - 1)
        } else {
            original_size
        };

        return aligned_size;
    }

    pub unsafe fn cleanup(&mut self) {
        self.device.device_wait_idle().unwrap();

        self.device
            .destroy_descriptor_set_layout(self.global_set_layout, None);
        self.device
            .destroy_descriptor_set_layout(self.mesh_shader_data.descriptor_set_layout, None);
        self.device
            .destroy_descriptor_set_layout(self.object_set_layout, None);
        self.device
            .destroy_descriptor_pool(self.descriptor_pool, None);

        vk_initializers::free_buffer_and_memory(
            &self.device,
            self.scene_data_buffer.buffer,
            self.scene_data_buffer.buffer_memory,
        );

        for buffer in self.mesh_shader_data.buffers.iter() {
            vk_initializers::free_buffer_and_memory(
                &self.device,
                buffer.buffer,
                buffer.buffer_memory,
            );
        }

        self.device.destroy_command_pool(self.command_pool, None);

        for frame_data in self.frame_data.iter() {
            self.device
                .destroy_semaphore(frame_data.present_semaphore, None);
            self.device
                .destroy_semaphore(frame_data.render_semaphore, None);
            self.device.destroy_fence(frame_data.render_fence, None);

            vk_initializers::free_buffer_and_memory(
                &self.device,
                frame_data.object_buffer.buffer,
                frame_data.object_buffer.buffer_memory,
            );
        }
        for &framebuffer in self.framebuffers.iter() {
            self.device.destroy_framebuffer(framebuffer, None);
        }

        for compute_pipeline in self.compute_pipelines.iter() {
            self.device.destroy_pipeline(*compute_pipeline, None);
        }

        for (_, material) in self.materials.iter() {
            self.device.destroy_pipeline(material.pipeline, None);
            self.device
                .destroy_pipeline_layout(material.pipeline_layout, None);
        }
        self.device.destroy_render_pass(self.render_pass, None);

        self.device.destroy_image_view(self.depth_image_view, None);
        self.device.destroy_image(self.depth_image.image, None);
        self.device.free_memory(self.depth_image.image_memory, None);

        for &image_view in self.swapchain_image_views.iter() {
            self.device.destroy_image_view(image_view, None);
        }

        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);

        self.device.destroy_device(None);
        self.surface_loader.destroy_surface(self.surface, None);

        self.debug_utils_loader
            .destroy_debug_utils_messenger(self.debug_callback, None);

        self.instance.destroy_instance(None);
    }

    pub unsafe fn draw(&mut self) {
        let frame_index = self.frame_count as usize % FRAME_OVERLAP;
        let frame_data = &self.frame_data[frame_index];

        self.device
            .wait_for_fences(&[frame_data.render_fence], true, std::u64::MAX)
            .unwrap();
        self.device
            .reset_fences(&[frame_data.render_fence])
            .unwrap();

        let swapchain_image_index = self
            .swapchain_loader
            .acquire_next_image(
                self.swapchain,
                std::u64::MAX,
                frame_data.present_semaphore,
                vk::Fence::null(),
            )
            .unwrap()
            .0;

        self.device
            .reset_command_buffer(
                frame_data.command_buffer,
                vk::CommandBufferResetFlags::from_raw(0),
            )
            .unwrap();

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device
            .begin_command_buffer(frame_data.command_buffer, &command_buffer_begin_info)
            .unwrap();

        let frame = self.frame_count as f32 / 120.0f32;
        let view = Mat4::look_at_rh(
            self.camera.pos,
            self.camera.pos + self.camera.front,
            self.camera.up,
        );
        let projection = Mat4::prespective(
            self.camera.fov,
            self.size.width as f32 / self.size.height as f32,
            0.1,
            1000.0,
        );
        let mut scene_data: SceneData = std::mem::zeroed();
        scene_data.view = view;
        scene_data.projection = projection;
        scene_data.fog_distances.x = L_X;
        scene_data.fog_distances.y = L_Y;
        scene_data.fog_distances.z = std::time::Instant::now()
            .duration_since(self.start)
            .as_millis() as f32
            / 1000.0;
        scene_data.fog_distances.w = self.camera.fov;

        let scene_data_offset =
            (Self::return_aligned_size(self.physical_device_properties, size_of::<SceneData>())
                * frame_index) as u64;
        let scene_data_ptr = self
            .device
            .map_memory(
                self.scene_data_buffer.buffer_memory,
                scene_data_offset,
                size_of::<SceneData>() as u64,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap() as *mut SceneData;
        scene_data_ptr.copy_from_nonoverlapping([scene_data].as_ptr(), 1);
        self.device
            .unmap_memory(self.scene_data_buffer.buffer_memory);

        // Spectrum and row ifft phase
        self.device.cmd_bind_pipeline(
            frame_data.command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.compute_pipelines[0],
        );
        self.device.cmd_bind_descriptor_sets(
            frame_data.command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.materials["ocean"].pipeline_layout,
            0,
            &[
                self.global_descriptor_set,
                self.mesh_shader_data.descriptor_set,
            ],
            &[scene_data_offset as u32],
        );
        self.device.cmd_push_constants(
            frame_data.command_buffer,
            self.materials["ocean"].pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &[Vec4 {
                x: 0.0f32,
                y: 0.0f32,
                z: 0.0f32,
                w: 0.0f32,
            }]
            .align_to::<u8>()
            .1, // Forgive me, father, for I have sinned.
        );
        self.device
            .cmd_dispatch(frame_data.command_buffer, OCEAN_PATCH_DIM as u32, 1, 1);

        let memory_barrier = vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build();
        let memory_barriers = [memory_barrier];
        self.device.cmd_pipeline_barrier(
            frame_data.command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::DEVICE_GROUP,
            &memory_barriers,
            &[],
            &[],
        );

        // Column ifft phase
        self.device.cmd_push_constants(
            frame_data.command_buffer,
            self.materials["ocean"].pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &[Vec4 {
                x: 1.0f32,
                y: 0.0f32,
                z: 0.0f32,
                w: 0.0f32,
            }]
            .align_to::<u8>()
            .1, // Forgive me, father, for I have sinned.
        );
        self.device
            .cmd_dispatch(frame_data.command_buffer, OCEAN_PATCH_DIM as u32, 1, 1);

        let memory_barrier = vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build();
        let memory_barriers = [memory_barrier];
        self.device.cmd_pipeline_barrier(
            frame_data.command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::MESH_SHADER_NV,
            vk::DependencyFlags::DEVICE_GROUP,
            &memory_barriers,
            &[],
            &[],
        );

        let mut clear_value = vk::ClearValue::default();
        clear_value.color = vk::ClearColorValue {
            float32: [
                15.0f32 / 256.0f32,
                15.0f32 / 256.0f32,
                15.0f32 / 256.0f32,
                1.0f32,
            ],
        };

        let mut depth_clear_value = vk::ClearValue::default();
        depth_clear_value.depth_stencil.depth = 1.0f32;

        let clear_values = [clear_value, depth_clear_value];
        let renderpass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.size,
            })
            .framebuffer(self.framebuffers[swapchain_image_index as usize])
            .clear_values(&clear_values);
        self.device.cmd_begin_render_pass(
            frame_data.command_buffer,
            &renderpass_begin_info,
            vk::SubpassContents::INLINE,
        );

        self.device.cmd_bind_pipeline(
            frame_data.command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.materials["skybox"].pipeline,
        );

        self.device.cmd_bind_descriptor_sets(
            frame_data.command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.materials["skybox"].pipeline_layout,
            0,
            &[self.global_descriptor_set],
            &[scene_data_offset as u32],
        );

        self.mesh_shader
            .cmd_draw_mesh_tasks(frame_data.command_buffer, 1, 0);

        self.device.cmd_bind_pipeline(
            frame_data.command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.materials["ocean"].pipeline,
        );

        self.device.cmd_bind_descriptor_sets(
            frame_data.command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.materials["ocean"].pipeline_layout,
            0,
            &[
                self.global_descriptor_set,
                self.mesh_shader_data.descriptor_set,
            ],
            &[scene_data_offset as u32],
        );

        self.mesh_shader.cmd_draw_mesh_tasks(
            frame_data.command_buffer,
            (OCEAN_PATCH_DIM * OCEAN_PATCH_DIM) as u32 / (16 * 16),
            0,
        );

        self.device.cmd_end_render_pass(frame_data.command_buffer);
        self.device
            .end_command_buffer(frame_data.command_buffer)
            .unwrap();

        let submit_info = vk::SubmitInfo::builder()
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .wait_semaphores(&[frame_data.present_semaphore])
            .signal_semaphores(&[frame_data.render_semaphore])
            .command_buffers(&[frame_data.command_buffer])
            .build();
        let submit_infos = [submit_info];
        self.device
            .queue_submit(self.graphics_queue, &submit_infos, frame_data.render_fence)
            .unwrap();

        let swapchains = [self.swapchain];
        let wait_semaphores = [frame_data.render_semaphore];
        let image_indices = [swapchain_image_index];
        let present_info = vk::PresentInfoKHR::builder()
            .swapchains(&swapchains)
            .wait_semaphores(&wait_semaphores)
            .image_indices(&image_indices);
        self.swapchain_loader
            .queue_present(self.graphics_queue, &present_info)
            .unwrap();

        self.frame_count += 1;
    }

    pub fn run(&mut self) {
        let mut event_pump = self.sdl_context.event_pump().unwrap();

        'running: loop {
            let current_timestamp = std::time::Instant::now();
            let delta_time = current_timestamp
                .duration_since(self.last_timestamp)
                .as_millis() as f32
                / 1000.0;
            self.last_timestamp = current_timestamp;

            for event in event_pump.poll_iter() {
                match event {
                    Event::Quit { .. }
                    | Event::KeyDown {
                        keycode: Some(Keycode::Escape),
                        ..
                    } => {
                        break 'running;
                    }
                    Event::MouseMotion {
                        xrel: x, yrel: y, ..
                    } => {
                        // Note: I'm not sure if xrel and yrel account for deltas between frames.
                        let sensitivity = 0.1;
                        self.camera.yaw += sensitivity * x as f32;
                        self.camera.pitch =
                            (self.camera.pitch + sensitivity * y as f32).clamp(-89.0, 89.0);
                        self.camera.front = Vec3 {
                            x: self.camera.yaw.to_radians().cos(),
                            y: self.camera.pitch.to_radians().sin(),
                            z: self.camera.yaw.to_radians().sin()
                                * self.camera.pitch.to_radians().cos(),
                        }
                    }
                    Event::MouseWheel { y: scroll_y, .. } => {
                        self.camera.fov = (self.camera.fov - scroll_y as f32).clamp(1.0, 60.0);
                    }
                    Event::KeyDown {
                        keycode: Some(pressed_key),
                        ..
                    } => {
                        let camera_speed = 100.0 * delta_time;
                        if pressed_key == Keycode::W {
                            self.camera.pos += self.camera.front * camera_speed;
                        }
                        if pressed_key == Keycode::S {
                            self.camera.pos -= self.camera.front * camera_speed;
                        }
                        if pressed_key == Keycode::A {
                            self.camera.pos -= Vec3::cross(self.camera.front, self.camera.up)
                                .normal()
                                * camera_speed;
                        }
                        if pressed_key == Keycode::D {
                            self.camera.pos += Vec3::cross(self.camera.front, self.camera.up)
                                .normal()
                                * camera_speed;
                        }
                    }
                    _ => {}
                }
            }

            unsafe {
                self.draw();
            };
        }
    }
}
