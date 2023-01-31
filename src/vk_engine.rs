extern crate ash;
extern crate imgui;
extern crate sdl2;

use crate::camera::Camera;
use crate::math::fft::Complex;
use crate::math::lin_alg::{Mat4, Vec2, Vec3, Vec4};
use crate::math::rand;
use crate::{imgui_backend, vk_helpers::*};

use ash::extensions::{
    ext::DebugUtils,
    khr::{Surface, Swapchain},
};
use ash::{vk, Device, Instance};
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::video::Window;
use std::collections::HashMap;
use std::mem::size_of;

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
    camera_pos: Vec4,
    fog_distances: Vec4,
    ambient_color: Vec4,
    sun_light_dir: Vec4,
    sun_light_col: Vec4,
}

struct OceanParams {
    L: f32,
    U: f32,
    F: f32,
    h: f32,
    ocean_dim: u32,
    noise_and_wavenumber_tex_idx: u32,
    waves_spectrum_idx: u32,
}

unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::std::slice::from_raw_parts((p as *const T) as *const u8, ::std::mem::size_of::<T>())
}

const FRAME_OVERLAP: usize = 2;
const OCEAN_PATCH_DIM: usize = 512;
// Only works for powers of 2
const OCEAN_PATCH_DIM_EXPONENT: u8 = (OCEAN_PATCH_DIM - 1).count_ones() as u8;
const OCEAN_PATCH_DIM_RECIPROCAL: f32 = 1.0 / (OCEAN_PATCH_DIM as f32);
const L: f32 = 1.0;
const TWO_PI: f32 = std::f32::consts::PI * 2.0;
const MAX_BINDLESS_COUNT: u32 = 2048;

const OCEAN_GRID_SHADER_SRC: &str = include_str!(".././assets/shaders/ocean_grid.comp.hlsl");
const OCEAN_SHADER_SRC: &str = include_str!(".././assets/shaders/ocean.comp.vert.frag.hlsl");
const SKYBOX_SHADER_SRC: &str = include_str!(".././assets/shaders/skybox.vert.frag.hlsl");
const INITIAL_SPECTRUM_SRC: &str = include_str!(".././assets/shaders/initial_spectrum.comp.hlsl");

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
    compute_pipelines: Vec<vk::Pipeline>,
    start: std::time::Instant,
    textures: HashMap<String, VkTexture>,
    imgui_ctx: imgui::Context,
    imgui_renderer: imgui_backend::Renderer,
    time_factor: f32,
    choppiness: f32,
    waves_descriptor_set: vk::DescriptorSet,
    waves_descriptor_set_layout: vk::DescriptorSetLayout,
    ocean_grid_vertex_buffer: VkBuffer,
    ocean_grid_index_buffer: VkBuffer,
    bindless_descriptor_pool: vk::DescriptorPool,
    bindless_textures_descriptor_set: vk::DescriptorSet,
    bindless_storage_images_descriptor_set: vk::DescriptorSet,
    initial_spectrum_creation_layout: vk::PipelineLayout,
    ocean_params: OceanParams,
}

impl VkEngine {
    pub fn new(width: u32, height: u32) -> VkEngine {
        let (sdl_context, window) = create_sdl_window(width, height);

        let (entry, instance) = create_instance(&window);

        let (debug_utils_loader, debug_callback) = create_debug_layer(&entry, &instance);

        let (surface, surface_loader) = create_surface(&window, &entry, &instance);

        let (physical_device, queue_family_index) =
            get_physical_device_and_graphics_queue_family_index(
                &instance,
                &surface_loader,
                &surface,
            );

        let physical_device_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };

        let device = create_device(queue_family_index, &instance, &physical_device);

        let surface_format = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()[0]
        };

        let (swapchain_loader, swapchain) = create_swapchain_loader_and_swapchain(
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
        let swapchain_image_views =
            create_swapchain_image_views(&swapchain_images, &surface_format, &device);

        let depth_image_extent = vk::Extent3D::builder()
            .width(width)
            .height(height)
            .depth(1)
            .build();
        let depth_image_format = vk::Format::D32_SFLOAT;

        let depth_image_create_info = create_image_create_info(
            depth_image_format,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            depth_image_extent,
        );
        let depth_image = create_image(
            &instance,
            physical_device,
            &device,
            depth_image_create_info,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        let depth_image_view_create_info = create_image_view_create_info(
            depth_image_format,
            depth_image.image,
            vk::ImageAspectFlags::DEPTH,
        );
        let depth_image_view = unsafe {
            device
                .create_image_view(&depth_image_view_create_info, None)
                .unwrap()
        };

        let render_pass = create_renderpass(&surface_format, &device);

        let framebuffers = create_framebuffers(
            &swapchain_image_views,
            depth_image_view,
            &render_pass,
            width,
            height,
            &device,
        );

        let graphics_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let scene_buffer_binding = descriptor_set_layout_binding(
            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            0,
            1,
            vk::ShaderStageFlags::COMPUTE
                | vk::ShaderStageFlags::VERTEX
                | vk::ShaderStageFlags::FRAGMENT,
        );

        let bindings = [scene_buffer_binding];
        let global_set_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        let global_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&global_set_info, None)
                .unwrap()
        };

        let object_buffer_binding = descriptor_set_layout_binding(
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
                .descriptor_count(MAX_BINDLESS_COUNT)
                .build(),
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(MAX_BINDLESS_COUNT)
                .build(),
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(MAX_BINDLESS_COUNT)
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

        let pool_sizes = [
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(MAX_BINDLESS_COUNT)
                .build(),
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(MAX_BINDLESS_COUNT)
                .build(),
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(MAX_BINDLESS_COUNT)
                .build(),
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(MAX_BINDLESS_COUNT)
                .build(),
        ];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(10)
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);
        let bindless_descriptor_pool = unsafe {
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

        let scene_param_buffer_size =
            FRAME_OVERLAP * return_aligned_size(physical_device_properties, size_of::<SceneData>());
        let VkBuffer {
            buffer: scene_param_buffer,
            buffer_memory: scene_param_buffer_memory,
        } = create_buffer(
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

        let (command_pool, command_buffers) =
            create_command_pool_and_buffer(queue_family_index, &device, FRAME_OVERLAP as u32);
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

        let ocean_grid_vertex_buffer = create_buffer(
            &instance,
            physical_device,
            &device,
            (size_of::<Vec4>() * OCEAN_PATCH_DIM * OCEAN_PATCH_DIM) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let ocean_grid_index_buffer = create_buffer(
            &instance,
            physical_device,
            &device,
            (size_of::<u32>() * (OCEAN_PATCH_DIM - 1) * (OCEAN_PATCH_DIM - 1) * 6) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        let ocean_grid_vertex_buffer_binding = descriptor_set_layout_binding(
            vk::DescriptorType::STORAGE_BUFFER,
            0,
            1,
            vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX,
        );
        let ocean_grid_index_buffer_binding = descriptor_set_layout_binding(
            vk::DescriptorType::STORAGE_BUFFER,
            1,
            1,
            vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX,
        );
        let bindings = [
            ocean_grid_vertex_buffer_binding,
            ocean_grid_index_buffer_binding,
        ];
        let ocean_grid_descriptor_set_layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        let ocean_grid_descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&ocean_grid_descriptor_set_layout_create_info, None)
                .unwrap()
        };
        let descriptor_set_layouts = [ocean_grid_descriptor_set_layout];
        let ocean_grid_descriptor_set = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&descriptor_set_layouts),
                )
                .unwrap()[0]
        };
        let ocean_grid_vertex_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(ocean_grid_vertex_buffer.buffer)
            .range(vk::WHOLE_SIZE)
            .build();
        let buffer_infos = [ocean_grid_vertex_buffer_info];
        let ocean_grid_vertex_write_descriptor = vk::WriteDescriptorSet::builder()
            .dst_set(ocean_grid_descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_infos)
            .build();
        let ocean_grid_index_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(ocean_grid_index_buffer.buffer)
            .range(vk::WHOLE_SIZE)
            .build();
        let buffer_infos = [ocean_grid_index_buffer_info];
        let ocean_grid_index_write_descriptor = vk::WriteDescriptorSet::builder()
            .dst_set(ocean_grid_descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_infos)
            .build();
        let descriptor_writes = [
            ocean_grid_vertex_write_descriptor,
            ocean_grid_index_write_descriptor,
        ];
        unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) };

        let shader_args = vec!["-spirv", "-Zi", "-Od"];
        let ocean_dim = OCEAN_PATCH_DIM.to_string();
        let ocean_dim_exponent = OCEAN_PATCH_DIM_EXPONENT.to_string();
        let ocean_dim_reciprocal = OCEAN_PATCH_DIM_RECIPROCAL.to_string();
        let shader_defines = vec![
            ("OCEAN_DIM", Some(ocean_dim.as_str())),
            ("OCEAN_DIM_EXPONENT", Some(ocean_dim_exponent.as_str())),
            ("OCEAN_DIM_RECIPROCAL", Some(ocean_dim_reciprocal.as_str())),
        ];

        let ocean_grid_shader_module = create_shader_module(
            &device,
            OCEAN_GRID_SHADER_SRC,
            "ocean_grid",
            "create_ocean_grid",
            "cs_6_5",
            &shader_args,
            &shader_defines,
        );

        let ocean_grid_compute_pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layouts),
                    None,
                )
                .unwrap()
        };

        let ocean_grid_compute_pipeline_create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(pipeline_shader_stage_create_info(
                vk::ShaderStageFlags::COMPUTE,
                ocean_grid_shader_module,
                "create_ocean_grid\0",
            ))
            .layout(ocean_grid_compute_pipeline_layout)
            .build();
        let compute_infos = [ocean_grid_compute_pipeline_create_info];
        let ocean_grid_compute_pipeline = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &compute_infos, None)
                .unwrap()[0]
        };

        unsafe {
            device.reset_fences(&[frame_data[0].render_fence]).unwrap();

            device
                .reset_command_buffer(command_buffers[0], vk::CommandBufferResetFlags::from_raw(0))
                .unwrap();

            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(command_buffers[0], &command_buffer_begin_info)
                .unwrap();

            device.cmd_bind_pipeline(
                command_buffers[0],
                vk::PipelineBindPoint::COMPUTE,
                ocean_grid_compute_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffers[0],
                vk::PipelineBindPoint::COMPUTE,
                ocean_grid_compute_pipeline_layout,
                0,
                &[ocean_grid_descriptor_set],
                &[],
            );
            device.cmd_dispatch(
                command_buffers[0],
                (OCEAN_PATCH_DIM / 16) as u32,
                (OCEAN_PATCH_DIM / 16) as u32,
                1,
            );
            device.end_command_buffer(command_buffers[0]).unwrap();
            device
                .queue_submit(
                    graphics_queue,
                    &[vk::SubmitInfo::builder()
                        .command_buffers(&[command_buffers[0]])
                        .build()],
                    frame_data[0].render_fence,
                )
                .unwrap();
            device.device_wait_idle().unwrap();

            device.destroy_descriptor_set_layout(ocean_grid_descriptor_set_layout, None);
            device
                .free_descriptor_sets(descriptor_pool, &[ocean_grid_descriptor_set])
                .unwrap();
            device.destroy_pipeline_layout(ocean_grid_compute_pipeline_layout, None);
            device.destroy_pipeline(ocean_grid_compute_pipeline, None);
            device.destroy_shader_module(ocean_grid_shader_module, None);
        }

        let mut extra_layout_info = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::builder()
            .binding_flags(&[vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                | vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT])
            .build();
        let descriptor_binding = [vk::DescriptorSetLayoutBinding::builder()
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(MAX_BINDLESS_COUNT)
            .binding(0)
            .stage_flags(vk::ShaderStageFlags::ALL)
            .build()];
        let bindless_textures_layout_create_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .bindings(&descriptor_binding)
            .push_next(&mut extra_layout_info);
        let bindless_textures_layout = unsafe {
            device
                .create_descriptor_set_layout(&bindless_textures_layout_create_info, None)
                .unwrap()
        };
        let mut extra_layout_info = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::builder()
            .binding_flags(&[vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                | vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT])
            .build();
        let descriptor_binding = [vk::DescriptorSetLayoutBinding::builder()
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(MAX_BINDLESS_COUNT)
            .binding(0)
            .stage_flags(vk::ShaderStageFlags::ALL)
            .build()];
        let bindless_storage_images_layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::builder()
                .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                .bindings(&descriptor_binding)
                .push_next(&mut extra_layout_info);
        let bindless_storage_images_layout = unsafe {
            device
                .create_descriptor_set_layout(&bindless_storage_images_layout_create_info, None)
                .unwrap()
        };
        let mut count_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo::builder()
            .descriptor_counts(&[MAX_BINDLESS_COUNT, MAX_BINDLESS_COUNT])
            .build();
        let descriptor_sets = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(bindless_descriptor_pool)
                        .set_layouts(&[bindless_textures_layout, bindless_storage_images_layout])
                        .push_next(&mut count_info)
                        .build(),
                )
                .unwrap()
        };
        let bindless_textures_descriptor_set = descriptor_sets[0];
        let bindless_storage_images_descriptor_set = descriptor_sets[1];

        let mut noise_and_wavenumber: Vec<Vec4> =
            vec![unsafe { std::mem::zeroed() }; OCEAN_PATCH_DIM * OCEAN_PATCH_DIM];
        let mut pcg_rng = rand::PCGRandom32::new();
        let start = OCEAN_PATCH_DIM as f32 / 2.0;

        for i in 0..OCEAN_PATCH_DIM {
            for j in 0..OCEAN_PATCH_DIM {
                let k = Vec2 {
                    x: TWO_PI * (start - j as f32) / L,
                    y: TWO_PI * (start - i as f32) / L,
                };

                let (u1, u2) = (
                    pcg_rng.next() as f32 / u32::MAX as f32,
                    pcg_rng.next() as f32 / u32::MAX as f32,
                );
                let (r1, r2) = rand::box_muller_rng(u1, u2);

                noise_and_wavenumber[i * OCEAN_PATCH_DIM + j] = Vec4 {
                    x: r1,
                    y: r2,
                    z: k.x,
                    w: k.y,
                };
            }
        }

        let noise_and_wavenumber_size = (size_of::<Vec4>() * noise_and_wavenumber.len()) as u64;
        let staging_buffer = create_buffer(
            &instance,
            physical_device,
            &device,
            noise_and_wavenumber_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data_ptr = device
                .map_memory(
                    staging_buffer.buffer_memory,
                    0,
                    noise_and_wavenumber_size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap() as *mut Vec4;
            data_ptr.copy_from_nonoverlapping(
                noise_and_wavenumber.as_ptr(),
                noise_and_wavenumber.len(),
            );
            device.unmap_memory(staging_buffer.buffer_memory);
        }

        let mut textures = HashMap::<String, VkTexture>::new();

        let noise_and_wavenumber_extent = vk::Extent3D {
            width: OCEAN_PATCH_DIM as u32,
            height: OCEAN_PATCH_DIM as u32,
            depth: 1,
        };
        let noise_and_wavenumber_info = add_texture(
            &instance,
            physical_device,
            &device,
            command_pool,
            graphics_queue,
            noise_and_wavenumber_extent,
            vk::Format::R32G32B32A32_SFLOAT,
            Some(staging_buffer),
            "noise_and_wavenumber",
            &mut textures,
        );
        let noise_and_wavenumber_write = vk::WriteDescriptorSet::builder()
            .dst_set(bindless_textures_descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&[noise_and_wavenumber_info])
            .build();

        let waves_extent = vk::Extent3D {
            width: OCEAN_PATCH_DIM as u32,
            height: OCEAN_PATCH_DIM as u32,
            depth: 1,
        };
        let waves_info = add_texture(
            &instance,
            physical_device,
            &device,
            command_pool,
            graphics_queue,
            waves_extent,
            vk::Format::R32G32B32A32_SFLOAT,
            None,
            "waves_2",
            &mut textures,
        );
        let waves_set_write = vk::WriteDescriptorSet::builder()
            .dst_set(bindless_storage_images_descriptor_set)
            .dst_array_element(0)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&[waves_info])
            .build();

        unsafe {
            device.update_descriptor_sets(&[noise_and_wavenumber_write, waves_set_write], &[]);
        };

        let initial_spectrum_creation_shader_module = create_shader_module(
            &device,
            INITIAL_SPECTRUM_SRC,
            "initial_spectrum",
            "create_initial_spectrum",
            "cs_6_5",
            &shader_args,
            &shader_defines,
        );

        let initial_spectrum_creation_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&[bindless_textures_layout, bindless_storage_images_layout])
            .push_constant_ranges(&[vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(28)
                .build()])
            .build();
        let initial_spectrum_creation_layout = unsafe {
            device
                .create_pipeline_layout(&initial_spectrum_creation_layout_info, None)
                .unwrap()
        };
        let initial_spectrum_creation_create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(pipeline_shader_stage_create_info(
                vk::ShaderStageFlags::COMPUTE,
                initial_spectrum_creation_shader_module,
                "create_initial_spectrum\0",
            ))
            .layout(initial_spectrum_creation_layout)
            .build();

        let mut waves: Vec<Vec4> =
            vec![unsafe { std::mem::zeroed() }; (OCEAN_PATCH_DIM + 1) * (OCEAN_PATCH_DIM + 1)];

        let amplitude = 0.45 * 1e-3;
        let wind_speed = 6.5;
        let wind_direction = Vec2 { x: -0.4, y: -0.9 };
        let l = wind_speed * wind_speed / 9.81;
        let l_minor_waves = l / 1000.0;

        for i in 0..=OCEAN_PATCH_DIM {
            for j in 0..=OCEAN_PATCH_DIM {
                let k = Vec2 {
                    x: TWO_PI * (start - j as f32) / L,
                    y: TWO_PI * (start - i as f32) / L,
                };
                let k_2 = k.length_sqr();
                let k_dot_w = Vec2::dot(k, wind_direction.normal());

                let b = if k.x != 0.0 || k.y != 0.0 {
                    f32::exp(-1.0 / (k_2 * l * l)) / (k_2 * k_2 * k_2)
                } else {
                    0.0
                };
                let c = f32::powi(k_dot_w, 2);

                let mut phillips_k = amplitude * b * c;
                if k_dot_w < 0.0 {
                    phillips_k *= 0.07;
                }
                phillips_k *= f32::exp(-k_2 * l_minor_waves * l_minor_waves);

                let h_zero_k = f32::sqrt(phillips_k) * std::f32::consts::FRAC_1_SQRT_2;

                let (u1, u2) = (
                    pcg_rng.next() as f32 / u32::MAX as f32,
                    pcg_rng.next() as f32 / u32::MAX as f32,
                );
                let (r1, r2) = rand::box_muller_rng(u1, u2);

                let tilde_h_zero = Complex { real: r1, imag: r2 } * h_zero_k;

                let idx = i * (OCEAN_PATCH_DIM + 1) + j;
                waves[idx] = Vec4 {
                    x: tilde_h_zero.real,
                    y: tilde_h_zero.imag,
                    z: k.x,
                    w: k.y,
                };
            }
        }

        let waves_binding = descriptor_set_layout_binding(
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            0,
            1,
            vk::ShaderStageFlags::COMPUTE,
        );
        let height_output_input_binding = descriptor_set_layout_binding(
            vk::DescriptorType::STORAGE_IMAGE,
            1,
            1,
            vk::ShaderStageFlags::COMPUTE,
        );
        let height_input_output_binding = descriptor_set_layout_binding(
            vk::DescriptorType::STORAGE_IMAGE,
            2,
            1,
            vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX,
        );
        let displacement_output_input_binding = descriptor_set_layout_binding(
            vk::DescriptorType::STORAGE_IMAGE,
            3,
            1,
            vk::ShaderStageFlags::COMPUTE,
        );
        let displacement_input_output_binding = descriptor_set_layout_binding(
            vk::DescriptorType::STORAGE_IMAGE,
            4,
            1,
            vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX,
        );

        let bindings = [
            waves_binding,
            height_output_input_binding,
            height_input_output_binding,
            displacement_output_input_binding,
            displacement_input_output_binding,
        ];
        let waves_descriptor_set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        let waves_descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&waves_descriptor_set_layout_info, None)
                .unwrap()
        };
        let set_layouts = [waves_descriptor_set_layout];
        let waves_descriptor_allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let waves_descriptor_set = unsafe {
            device
                .allocate_descriptor_sets(&waves_descriptor_allocate_info)
                .unwrap()[0]
        };

        // let waves_size = (size_of::<Vec4>() * waves.len()) as u64;
        // let staging_buffer = create_buffer(
        //     &instance,
        //     physical_device,
        //     &device,
        //     waves_size,
        //     vk::BufferUsageFlags::TRANSFER_SRC,
        //     vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        // );

        // unsafe {
        //     let data_ptr = device
        //         .map_memory(
        //             staging_buffer.buffer_memory,
        //             0,
        //             waves_size,
        //             vk::MemoryMapFlags::empty(),
        //         )
        //         .unwrap() as *mut Vec4;
        //     data_ptr.copy_from_nonoverlapping(waves.as_ptr(), waves.len());
        //     device.unmap_memory(staging_buffer.buffer_memory);
        // }

        // let waves_extent = vk::Extent3D {
        //     width: (OCEAN_PATCH_DIM + 1) as u32,
        //     height: (OCEAN_PATCH_DIM + 1) as u32,
        //     depth: 1,
        // };
        // let waves_info = add_texture(
        //     &instance,
        //     physical_device,
        //     &device,
        //     command_pool,
        //     graphics_queue,
        //     waves_extent,
        //     vk::Format::R32G32B32A32_SFLOAT,
        //     Some(staging_buffer),
        //     "waves",
        //     &mut textures,
        // );
        let infos = [waves_info];
        let waves_set_write = vk::WriteDescriptorSet::builder()
            .dst_set(waves_descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&infos)
            .build();

        let height_output_input_info = add_texture(
            &instance,
            physical_device,
            &device,
            command_pool,
            graphics_queue,
            waves_extent,
            vk::Format::R32G32B32A32_SFLOAT,
            None,
            "height_output_input",
            &mut textures,
        );
        let infos = [height_output_input_info];
        let height_output_input_set_write = vk::WriteDescriptorSet::builder()
            .dst_set(waves_descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&infos)
            .build();

        let height_input_output_info = add_texture(
            &instance,
            physical_device,
            &device,
            command_pool,
            graphics_queue,
            waves_extent,
            vk::Format::R32G32B32A32_SFLOAT,
            None,
            "height_input_output",
            &mut textures,
        );
        let infos = [height_input_output_info];
        let height_input_output_set_write = vk::WriteDescriptorSet::builder()
            .dst_set(waves_descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&infos)
            .build();

        let displacement_output_input_info = add_texture(
            &instance,
            physical_device,
            &device,
            command_pool,
            graphics_queue,
            waves_extent,
            vk::Format::R32G32B32A32_SFLOAT,
            None,
            "displacement_output_input",
            &mut textures,
        );
        let infos = [displacement_output_input_info];
        let displacement_output_input_set_write = vk::WriteDescriptorSet::builder()
            .dst_set(waves_descriptor_set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&infos)
            .build();

        let displacement_input_output_info = add_texture(
            &instance,
            physical_device,
            &device,
            command_pool,
            graphics_queue,
            waves_extent,
            vk::Format::R32G32B32A32_SFLOAT,
            None,
            "displacement_input_output",
            &mut textures,
        );
        let infos = [displacement_input_output_info];
        let displacement_input_output_set_write = vk::WriteDescriptorSet::builder()
            .dst_set(waves_descriptor_set)
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&infos)
            .build();

        unsafe {
            device.update_descriptor_sets(
                &[
                    waves_set_write,
                    height_output_input_set_write,
                    height_input_output_set_write,
                    displacement_output_input_set_write,
                    displacement_input_output_set_write,
                ],
                &[],
            )
        };

        let descriptor_set_layouts = [global_set_layout, waves_descriptor_set_layout];
        let ocean_pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layouts),
                    None,
                )
                .unwrap()
        };

        let ocean_vert_shader_module = create_shader_module(
            &device,
            OCEAN_SHADER_SRC,
            "ocean",
            "vs_main",
            "vs_6_5",
            &shader_args,
            &shader_defines,
        );
        let ocean_fragment_shader_module = create_shader_module(
            &device,
            OCEAN_SHADER_SRC,
            "ocean",
            "fs_main",
            "ps_6_5",
            &shader_args,
            &shader_defines,
        );

        let mut pipeline_builder = VkPipelineBuilder::default();
        pipeline_builder
            .shader_stages
            .push(pipeline_shader_stage_create_info(
                vk::ShaderStageFlags::VERTEX,
                ocean_vert_shader_module,
                "vs_main\0",
            ));
        pipeline_builder
            .shader_stages
            .push(pipeline_shader_stage_create_info(
                vk::ShaderStageFlags::FRAGMENT,
                ocean_fragment_shader_module,
                "fs_main\0",
            ));
        pipeline_builder.vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&[vk::VertexInputBindingDescription::builder()
                .binding(0)
                .stride(size_of::<Vec4>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX)
                .build()])
            .vertex_attribute_descriptions(&[vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(0)
                .build()])
            .build();
        pipeline_builder.input_assembly =
            input_assembly_create_info(&vk::PrimitiveTopology::TRIANGLE_LIST);

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

        pipeline_builder.rasterizer = rasterization_state_create_info(vk::PolygonMode::FILL);

        pipeline_builder.multisampling = multisampling_state_create_info();

        pipeline_builder.color_blend_attachment = color_blend_attachment_state();

        pipeline_builder.depth_stencil_state = create_depth_stencil_create_info();

        pipeline_builder.pipeline_layout = ocean_pipeline_layout;

        let ocean_pipeline = pipeline_builder.build_pipline(&render_pass, &device);

        unsafe {
            device.destroy_shader_module(ocean_vert_shader_module, None);
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

        let skybox_vertex_shader_module = create_shader_module(
            &device,
            SKYBOX_SHADER_SRC,
            "skybox",
            "vs_main",
            "vs_6_5",
            &shader_args,
            &vec![],
        );
        let skybox_fragment_shader_module = create_shader_module(
            &device,
            SKYBOX_SHADER_SRC,
            "skybox",
            "fs_main",
            "ps_6_5",
            &shader_args,
            &vec![],
        );

        pipeline_builder.shader_stages.clear();
        pipeline_builder
            .shader_stages
            .push(pipeline_shader_stage_create_info(
                vk::ShaderStageFlags::VERTEX,
                skybox_vertex_shader_module,
                "vs_main\0",
            ));
        pipeline_builder
            .shader_stages
            .push(pipeline_shader_stage_create_info(
                vk::ShaderStageFlags::FRAGMENT,
                skybox_fragment_shader_module,
                "fs_main\0",
            ));
        pipeline_builder.vertex_input_info = vertex_input_state_create_info();
        pipeline_builder.input_assembly =
            input_assembly_create_info(&vk::PrimitiveTopology::TRIANGLE_LIST);

        pipeline_builder.rasterizer = rasterization_state_create_info(vk::PolygonMode::FILL);

        pipeline_builder.depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default();

        pipeline_builder.pipeline_layout = skybox_pipeline_layout;

        let skybox_pipeline = pipeline_builder.build_pipline(&render_pass, &device);

        unsafe {
            device.destroy_shader_module(skybox_vertex_shader_module, None);
            device.destroy_shader_module(skybox_fragment_shader_module, None);
        };

        let mut spectrum_shader_defines = shader_defines.clone();
        spectrum_shader_defines.push(("CALCULATE_SPECTRUM_AND_ROW_IFFT", None));
        let spectrum_and_row_ifft_shader_module = create_shader_module(
            &device,
            OCEAN_SHADER_SRC,
            "ocean",
            "cs_main",
            "cs_6_5",
            &shader_args,
            &spectrum_shader_defines,
        );
        let spectrum_and_row_ifft_pipline_create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(pipeline_shader_stage_create_info(
                vk::ShaderStageFlags::COMPUTE,
                spectrum_and_row_ifft_shader_module,
                "cs_main\0",
            ))
            .layout(ocean_pipeline_layout)
            .build();

        let column_ifft_shader_module = create_shader_module(
            &device,
            OCEAN_SHADER_SRC,
            "ocean",
            "cs_main",
            "cs_6_5",
            &shader_args,
            &shader_defines,
        );
        let column_ifft_pipline_create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(pipeline_shader_stage_create_info(
                vk::ShaderStageFlags::COMPUTE,
                column_ifft_shader_module,
                "cs_main\0",
            ))
            .layout(ocean_pipeline_layout)
            .build();

        let compute_infos = [
            initial_spectrum_creation_create_info,
            spectrum_and_row_ifft_pipline_create_info,
            column_ifft_pipline_create_info,
        ];
        let compute_pipelines = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &compute_infos, None)
                .unwrap()
        };

        unsafe {
            device.destroy_descriptor_set_layout(bindless_textures_layout, None);
            device.destroy_descriptor_set_layout(bindless_storage_images_layout, None);
            device.destroy_shader_module(initial_spectrum_creation_shader_module, None);
            device.destroy_shader_module(spectrum_and_row_ifft_shader_module, None);
            device.destroy_shader_module(column_ifft_shader_module, None);
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

        let mut imgui_ctx = imgui_backend::create_platform(&window);

        let imgui_renderer = imgui_backend::Renderer::init_renderer(
            &mut imgui_ctx,
            &instance,
            physical_device,
            &device,
            command_pool,
            graphics_queue,
            descriptor_pool,
            &mut textures,
            &shader_args,
            &mut pipeline_builder,
            render_pass,
        );

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
            compute_pipelines,
            start: std::time::Instant::now(),
            textures,
            imgui_ctx,
            imgui_renderer,
            time_factor: 1.0,
            choppiness: 0.0,
            ocean_grid_vertex_buffer,
            ocean_grid_index_buffer,
            waves_descriptor_set,
            waves_descriptor_set_layout,
            bindless_descriptor_pool,
            bindless_textures_descriptor_set,
            bindless_storage_images_descriptor_set,
            initial_spectrum_creation_layout,
            ocean_params: OceanParams {
                L: 1.0,
                U: 0.5,
                F: 100000.0,
                h: 500.0,
                ocean_dim: OCEAN_PATCH_DIM as u32,
                noise_and_wavenumber_tex_idx: 0,
                waves_spectrum_idx: 0,
            },
        };
    }

    pub unsafe fn cleanup(&mut self) {
        self.device.device_wait_idle().unwrap();

        self.imgui_renderer.deinit_renderer(&self.device);

        self.device
            .destroy_descriptor_pool(self.bindless_descriptor_pool, None);

        self.device
            .destroy_descriptor_set_layout(self.global_set_layout, None);
        self.device
            .destroy_descriptor_set_layout(self.waves_descriptor_set_layout, None);
        self.device
            .destroy_descriptor_set_layout(self.object_set_layout, None);
        self.device
            .destroy_descriptor_pool(self.descriptor_pool, None);

        free_buffer_and_memory(&self.device, &self.scene_data_buffer);
        free_buffer_and_memory(&self.device, &self.ocean_grid_vertex_buffer);
        free_buffer_and_memory(&self.device, &self.ocean_grid_index_buffer);

        self.device.destroy_command_pool(self.command_pool, None);

        for frame_data in self.frame_data.iter() {
            self.device
                .destroy_semaphore(frame_data.present_semaphore, None);
            self.device
                .destroy_semaphore(frame_data.render_semaphore, None);
            self.device.destroy_fence(frame_data.render_fence, None);

            free_buffer_and_memory(&self.device, &frame_data.object_buffer);
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

        self.device
            .destroy_pipeline_layout(self.initial_spectrum_creation_layout, None);

        self.device.destroy_image_view(self.depth_image_view, None);
        self.device.destroy_image(self.depth_image.image, None);
        self.device.free_memory(self.depth_image.image_memory, None);

        for &image_view in self.swapchain_image_views.iter() {
            self.device.destroy_image_view(image_view, None);
        }

        for (
            _,
            VkTexture {
                image,
                image_view,
                sampler,
            },
        ) in self.textures.iter()
        {
            self.device.destroy_sampler(*sampler, None);
            self.device.destroy_image_view(*image_view, None);
            self.device.destroy_image(image.image, None);
            self.device.free_memory(image.image_memory, None);
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
        let ui = self.imgui_ctx.frame();
        imgui::Slider::new("Time factor", 0.0, 1.0).build(&ui, &mut self.time_factor);
        imgui::Slider::new("Choppiness", -10.0, 0.0).build(&ui, &mut self.choppiness);
        imgui::Slider::new("Length", 0.0, 20.0).build(&ui, &mut self.ocean_params.L);
        imgui::Slider::new("Wind speed", 0.0, 20.0).build(&ui, &mut self.ocean_params.U);

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
        scene_data.camera_pos = Vec4::from_vec3(self.camera.pos, 0.0);
        scene_data.fog_distances.x = self.choppiness;
        scene_data.fog_distances.y = L;
        scene_data.fog_distances.z = std::time::Instant::now()
            .duration_since(self.start)
            .as_secs_f32()
            * self.time_factor;
        scene_data.fog_distances.w = self.camera.fov;
        scene_data.ambient_color.x = self.size.width as f32;
        scene_data.ambient_color.y = self.size.height as f32;

        let scene_data_offset =
            (return_aligned_size(self.physical_device_properties, size_of::<SceneData>())
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

        self.device.cmd_bind_pipeline(
            frame_data.command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.compute_pipelines[0],
        );
        self.device.cmd_bind_descriptor_sets(
            frame_data.command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.initial_spectrum_creation_layout,
            0,
            &[
                self.bindless_textures_descriptor_set,
                self.bindless_storage_images_descriptor_set,
            ],
            &[],
        );
        self.device.cmd_push_constants(
            frame_data.command_buffer,
            self.initial_spectrum_creation_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &unsafe { any_as_u8_slice(&self.ocean_params) },
        );
        self.device.cmd_dispatch(
            frame_data.command_buffer,
            (OCEAN_PATCH_DIM / 8) as u32,
            (OCEAN_PATCH_DIM / 8) as u32,
            1,
        );

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

        // Spectrum and row ifft phase
        self.device.cmd_bind_pipeline(
            frame_data.command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.compute_pipelines[1],
        );
        self.device.cmd_bind_descriptor_sets(
            frame_data.command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.materials["ocean"].pipeline_layout,
            0,
            &[self.global_descriptor_set, self.waves_descriptor_set],
            &[scene_data_offset as u32],
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
        self.device.cmd_bind_pipeline(
            frame_data.command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.compute_pipelines[2],
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
            vk::PipelineStageFlags::VERTEX_SHADER,
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

        self.device.cmd_draw(frame_data.command_buffer, 6, 1, 0, 0);

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
            &[self.global_descriptor_set, self.waves_descriptor_set],
            &[scene_data_offset as u32],
        );

        self.device.cmd_bind_vertex_buffers(
            frame_data.command_buffer,
            0,
            &[self.ocean_grid_vertex_buffer.buffer],
            &[0],
        );
        self.device.cmd_bind_index_buffer(
            frame_data.command_buffer,
            self.ocean_grid_index_buffer.buffer,
            0,
            vk::IndexType::UINT32,
        );

        self.device.cmd_draw_indexed(
            frame_data.command_buffer,
            ((OCEAN_PATCH_DIM - 1) * (OCEAN_PATCH_DIM - 1)) as u32 * 6,
            1,
            0,
            0,
            0,
        );

        let draw_data = ui.render();
        self.imgui_renderer
            .render(draw_data, &self.device, frame_data.command_buffer);

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
                .as_secs_f32();
            self.imgui_ctx
                .io_mut()
                .update_delta_time(current_timestamp - self.last_timestamp);
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

                    Event::MouseButtonDown {
                        mouse_btn: sdl2::mouse::MouseButton::Right,
                        ..
                    } => {
                        self.camera.rotate_camera = true;
                    }
                    Event::MouseButtonUp {
                        mouse_btn: sdl2::mouse::MouseButton::Right,
                        ..
                    } => self.camera.rotate_camera = false,

                    _ => {
                        self.camera.handle_event(&event, delta_time);
                        imgui_backend::handle_event(self.imgui_ctx.io_mut(), &event);
                    }
                }
            }

            unsafe {
                self.draw();
            };
        }
    }
}

impl Drop for VkEngine {
    fn drop(&mut self) {
        unsafe { self.cleanup() };
    }
}
