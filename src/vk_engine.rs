extern crate ash;
extern crate sdl2;

use crate::math::{Mat4, Vec3};
use crate::obj_loader::read_obj_file;
use crate::vk_initializers;

use ash::extensions::{
    ext::DebugUtils,
    khr::{Surface, Swapchain},
};
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::{vk, vk::PhysicalDevice, Device, Instance};
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

struct Vertex {
    pos: Vec3,
    norm: Vec3,
}

struct Mesh {
    vertices: Vec<Vertex>,
    vertex_buffer: VkBuffer,
    indices: Vec<u32>,
    index_buffer: VkBuffer,
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

const FRAME_OVERLAP: usize = 2;

struct FrameData {
    present_semaphore: vk::Semaphore,
    render_semaphore: vk::Semaphore,
    render_fence: vk::Fence,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
}

impl Vertex {
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        return vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build();
    }

    pub fn get_attribute_description() -> [vk::VertexInputAttributeDescription; 2] {
        return [
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(size_of::<Vec3>() as u32)
                .build(),
        ];
    }

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
        let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&[self.viewport])
            .scissors(&[self.scissor])
            .build();

        let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&[self.color_blend_attachment])
            .build();

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(self.shader_stages.as_slice())
            .vertex_input_state(
                &vk::PipelineVertexInputStateCreateInfo::builder()
                    .vertex_binding_descriptions(&[Vertex::get_binding_description()])
                    .vertex_attribute_descriptions(&Vertex::get_attribute_description())
                    .build(),
            )
            .input_assembly_state(&self.input_assembly)
            .viewport_state(&viewport_state_create_info)
            .rasterization_state(&self.rasterizer)
            .multisample_state(&self.multisampling)
            .color_blend_state(&color_blend_state_create_info)
            .depth_stencil_state(&self.depth_stencil_state)
            .layout(self.pipeline_layout)
            .render_pass(*render_pass)
            .build();
        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
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
    device: Device,
    swapchain_loader: Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_image_views: Vec<vk::ImageView>,
    depth_image: VkImage,
    depth_image_view: vk::ImageView,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    graphics_queue: vk::Queue,
    frame_data: [FrameData; FRAME_OVERLAP],
    triangle_vertex_shader_module: vk::ShaderModule,
    triangle_fragment_shader_module: vk::ShaderModule,
    meshes: HashMap<String, Mesh>,
    materials: HashMap<String, Material>,
    renderables: Vec<RenderObject>,
    camera: Camera,
    last_timestamp: std::time::Instant,
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

            let fence_create_info = vk::FenceCreateInfo::builder()
                .flags(vk::FenceCreateFlags::SIGNALED)
                .build();
            let render_fence = unsafe { device.create_fence(&fence_create_info, None).unwrap() };

            let (command_pool, command_buffers) =
                vk_initializers::create_command_pool_and_buffer(queue_family_index, &device);

            frame_data[i] = FrameData {
                present_semaphore,
                render_semaphore,
                render_fence,
                command_pool,
                command_buffer: command_buffers[0],
            };
        }

        let triangle_pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder()
                        .push_constant_ranges(&[vk::PushConstantRange::builder()
                            .size(size_of::<Mat4>() as u32 * 3)
                            .stage_flags(vk::ShaderStageFlags::VERTEX)
                            .build()])
                        .build(),
                    None,
                )
                .unwrap()
        };

        let (triangle_vertex_shader_module, triangle_fragment_shader_module) =
            vk_initializers::create_vertex_and_fragment_shader_modules(&device);

        let mut pipeline_builder = VkPipelineBuilder::default();

        pipeline_builder
            .shader_stages
            .push(vk_initializers::pipeline_shader_stage_create_info(
                vk::ShaderStageFlags::VERTEX,
                triangle_vertex_shader_module,
                "vs_main\0",
            ));
        pipeline_builder
            .shader_stages
            .push(vk_initializers::pipeline_shader_stage_create_info(
                vk::ShaderStageFlags::FRAGMENT,
                triangle_fragment_shader_module,
                "fs_main\0",
            ));

        pipeline_builder.vertex_input_info = vk_initializers::vertex_input_state_create_info();

        pipeline_builder.input_assembly =
            vk_initializers::input_assembly_create_info(&vk::PrimitiveTopology::TRIANGLE_LIST);

        pipeline_builder.viewport = vk::Viewport::builder()
            .x(0.0f32)
            .y(0.0f32)
            .width(width as f32)
            .height(height as f32)
            .min_depth(0.0f32)
            .max_depth(1.0f32)
            .build();

        pipeline_builder.scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(vk::Extent2D { width, height })
            .build();

        pipeline_builder.rasterizer =
            vk_initializers::rasterization_state_create_info(vk::PolygonMode::FILL);

        pipeline_builder.multisampling = vk_initializers::multisampling_state_create_info();

        pipeline_builder.color_blend_attachment = vk_initializers::color_blend_attachment_state();

        pipeline_builder.depth_stencil_state = vk_initializers::create_depth_stencil_create_info();

        pipeline_builder.pipeline_layout = triangle_pipeline_layout;

        let triangle_pipeline = pipeline_builder.build_pipline(&render_pass, &device);

        let monkey_mesh = VkEngine::load_and_upload_mesh(
            &instance,
            physical_device,
            &device,
            "./assets/models/monkey.obj",
        );

        let triangle_mesh = VkEngine::load_and_upload_mesh(
            &instance,
            physical_device,
            &device,
            "./assets/models/triangle.obj",
        );

        let camera = Camera::default();

        let mut mesh_map = HashMap::<String, Mesh>::new();
        mesh_map.insert("monkey".to_string(), monkey_mesh);
        mesh_map.insert("triangle".to_string(), triangle_mesh);

        let mut material_map = HashMap::<String, Material>::new();
        material_map.insert(
            "default".to_string(),
            Material {
                pipeline: triangle_pipeline,
                pipeline_layout: triangle_pipeline_layout,
            },
        );

        let mut scene = Vec::<RenderObject>::with_capacity(401);
        scene.push(RenderObject {
            mesh_key: "monkey".to_string(),
            material_key: "default".to_string(),
            transformation_matrix: Mat4::rotate(
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                180.0f32.to_radians(),
            ),
        });

        for x in -20..=20 {
            for y in -20..=20 {
                let translation = Mat4::translate(Vec3 {
                    x: x as f32,
                    y: 0.0,
                    z: y as f32,
                });
                let scale = Mat4::scale(Vec3 {
                    x: 0.2,
                    y: 0.2,
                    z: 0.2,
                });

                scene.push(RenderObject {
                    mesh_key: "triangle".to_string(),
                    material_key: "default".to_string(),
                    transformation_matrix: translation * scale,
                });
            }
        }

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
            device,
            swapchain_loader,
            swapchain,
            swapchain_image_views,
            depth_image,
            depth_image_view,
            render_pass,
            framebuffers,
            graphics_queue,
            frame_data,
            triangle_vertex_shader_module,
            triangle_fragment_shader_module,
            meshes: mesh_map,
            materials: material_map,
            renderables: scene,
            camera,
            last_timestamp: std::time::Instant::now(),
        };
    }

    fn load_and_upload_mesh(
        instance: &Instance,
        physical_device: PhysicalDevice,
        device: &Device,
        path: &str,
    ) -> Mesh {
        let (positions, normals, indices) = read_obj_file(path);
        let vertices = Vertex::construct_vertices_from_positions(positions, normals);

        let vertex_buffer_size = (size_of::<Vertex>() * vertices.len()) as u64;
        let index_buffer_size = (size_of::<u32>() * indices.len()) as u64;

        let (vertex_buffer, vertex_buffer_memory) = vk_initializers::create_buffer(
            &instance,
            physical_device,
            &device,
            vertex_buffer_size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        let (index_buffer, index_buffer_memory) = vk_initializers::create_buffer(
            &instance,
            physical_device,
            &device,
            index_buffer_size,
            vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            // Persistant mapping.
            let vertex_data_ptr = device
                .map_memory(
                    vertex_buffer_memory,
                    0,
                    vertex_buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap() as *mut Vertex;
            let index_data_ptr = device
                .map_memory(
                    index_buffer_memory,
                    0,
                    index_buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap() as *mut u32;

            vertex_data_ptr.copy_from_nonoverlapping(vertices.as_ptr(), vertices.len());
            index_data_ptr.copy_from_nonoverlapping(indices.as_ptr(), indices.len());
        };

        return Mesh {
            vertices,
            indices,
            vertex_buffer: VkBuffer {
                buffer: vertex_buffer,
                buffer_memory: vertex_buffer_memory,
            },
            index_buffer: VkBuffer {
                buffer: index_buffer,
                buffer_memory: index_buffer_memory,
            },
        };
    }

    pub unsafe fn cleanup(&mut self) {
        self.device.device_wait_idle().unwrap();

        self.device
            .destroy_shader_module(self.triangle_vertex_shader_module, None);
        self.device
            .destroy_shader_module(self.triangle_fragment_shader_module, None);

        for frame_data in self.frame_data.iter() {
            self.device
                .destroy_semaphore(frame_data.present_semaphore, None);
            self.device
                .destroy_semaphore(frame_data.render_semaphore, None);
            self.device.destroy_fence(frame_data.render_fence, None);

            self.device
                .destroy_command_pool(frame_data.command_pool, None);
        }
        for &framebuffer in self.framebuffers.iter() {
            self.device.destroy_framebuffer(framebuffer, None);
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

        for (_, mesh) in self.meshes.iter() {
            self.device.destroy_buffer(mesh.vertex_buffer.buffer, None);
            self.device.destroy_buffer(mesh.index_buffer.buffer, None);

            self.device
                .free_memory(mesh.vertex_buffer.buffer_memory, None);
            self.device
                .free_memory(mesh.index_buffer.buffer_memory, None);
        }

        self.device.destroy_device(None);
        self.surface_loader.destroy_surface(self.surface, None);

        self.debug_utils_loader
            .destroy_debug_utils_messenger(self.debug_callback, None);

        self.instance.destroy_instance(None);
    }

    pub unsafe fn draw(&mut self) {
        let frame_data = &self.frame_data[self.frame_count as usize % FRAME_OVERLAP];

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
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .build();
        self.device
            .begin_command_buffer(frame_data.command_buffer, &command_buffer_begin_info)
            .unwrap();

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

        let renderpass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.size,
            })
            .framebuffer(self.framebuffers[swapchain_image_index as usize])
            .clear_values(&[clear_value, depth_clear_value])
            .build();
        self.device.cmd_begin_render_pass(
            frame_data.command_buffer,
            &renderpass_begin_info,
            vk::SubpassContents::INLINE,
        );

        self.draw_objects();

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
        self.device
            .queue_submit(self.graphics_queue, &[submit_info], frame_data.render_fence)
            .unwrap();

        let present_info = vk::PresentInfoKHR::builder()
            .swapchains(&[self.swapchain])
            .wait_semaphores(&[frame_data.render_semaphore])
            .image_indices(&[swapchain_image_index])
            .build();
        self.swapchain_loader
            .queue_present(self.graphics_queue, &present_info)
            .unwrap();

        self.frame_count += 1;
    }

    unsafe fn draw_objects(&self) {
        let frame_data = &self.frame_data[self.frame_count as usize % FRAME_OVERLAP];

        let view = Mat4::look_at_rh(
            self.camera.pos,
            self.camera.pos + self.camera.front,
            self.camera.up,
        );

        let projection = Mat4::prespective(
            self.camera.fov,
            self.size.width as f32 / self.size.height as f32,
            0.1,
            100.0,
        );

        let (mut last_mesh_key, mut last_material_key) = ("".to_string(), "".to_string());
        for RenderObject {
            mesh_key,
            material_key,
            transformation_matrix,
        } in self.renderables.iter()
        {
            if *material_key != last_material_key {
                self.device.cmd_bind_pipeline(
                    frame_data.command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.materials[material_key].pipeline,
                );
                last_material_key = material_key.clone();
            }

            if *mesh_key != last_mesh_key {
                self.device.cmd_bind_vertex_buffers(
                    frame_data.command_buffer,
                    0,
                    &[self.meshes[mesh_key].vertex_buffer.buffer],
                    &[0],
                );
                self.device.cmd_bind_index_buffer(
                    frame_data.command_buffer,
                    self.meshes[mesh_key].index_buffer.buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                last_mesh_key = mesh_key.clone();
            }

            self.device.cmd_push_constants(
                frame_data.command_buffer,
                self.materials[material_key].pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                &[*transformation_matrix, view, projection]
                    .align_to::<u8>()
                    .1, // Forgive me, father, for I have sinned.
            );

            self.device.cmd_draw_indexed(
                frame_data.command_buffer,
                self.meshes[mesh_key].indices.len() as u32,
                1,
                0,
                0,
                0,
            );
        }
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
                        let camera_speed = 2.5 * delta_time;
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
