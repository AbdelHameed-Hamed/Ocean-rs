// Based on https://github.com/ocornut/imgui/blob/master/backends/imgui_impl_vulkan.cpp

use crate::vk_helpers::*;
use ash::{
    vk::{self, Handle},
    Device, Instance,
};
use std::mem::size_of;

const IMGUI_SHADER_SRC: &str = include_str!("./../assets/shaders/imgui.vert.frag.hlsl");

pub fn create_platform(window: &sdl2::video::Window) -> imgui::Context {
    let mut imgui_ctx = imgui::Context::create();
    imgui_ctx.set_ini_filename(None);
    let window_size = window.size();
    let drawable_size = window.drawable_size();
    imgui_ctx.io_mut().display_framebuffer_scale = [
        drawable_size.0 as f32 / window_size.0 as f32,
        drawable_size.1 as f32 / window_size.1 as f32,
    ];
    imgui_ctx.io_mut().display_size = [window_size.0 as f32, window_size.1 as f32];
    imgui_ctx.set_renderer_name(Some("Vulkan".to_string()));
    imgui_ctx.style_mut().use_dark_colors();

    return imgui_ctx;
}

pub fn handle_event(io: &mut imgui::Io, event: &sdl2::event::Event) {
    use sdl2::event::Event::*;
    use sdl2::mouse::*;

    match *event {
        MouseButtonDown { mouse_btn, .. } => match mouse_btn {
            MouseButton::Left => io.mouse_down[0] = true,
            MouseButton::Middle => io.mouse_down[2] = true,
            MouseButton::Right => io.mouse_down[1] = true,
            _ => {}
        },

        MouseButtonUp { mouse_btn, .. } => match mouse_btn {
            MouseButton::Left => io.mouse_down[0] = false,
            MouseButton::Middle => io.mouse_down[2] = false,
            MouseButton::Right => io.mouse_down[1] = false,
            _ => {}
        },

        MouseWheel { x, y, .. } => {
            io.mouse_wheel = y as f32;
            io.mouse_wheel_h = x as f32;
        }

        MouseMotion { x, y, .. } => io.mouse_pos = [x as f32, y as f32],

        _ => {}
    };
}

fn get_binding_description() -> vk::VertexInputBindingDescription {
    return vk::VertexInputBindingDescription::builder()
        .binding(0)
        .stride(size_of::<imgui::DrawVert>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX)
        .build();
}

fn get_attribute_description() -> [vk::VertexInputAttributeDescription; 3] {
    return [
        vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0)
            .build(),
        vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(size_of::<[f32; 2]>() as u32)
            .build(),
        vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R8G8B8A8_UNORM)
            .offset(size_of::<[f32; 4]>() as u32)
            .build(),
    ];
}

pub struct Renderer {
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    vert_buf: VkBuffer,
    idx_buf: VkBuffer,
}

impl Renderer {
    pub fn init_renderer(
        imgui_ctx: &mut imgui::Context,
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        device: &Device,
        command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
        descriptor_pool: vk::DescriptorPool,
        textures: &mut std::collections::HashMap<String, VkTexture>,
        shader_args: &Vec<&str>,
        pipeline_builder: &mut VkPipelineBuilder,
        render_pass: vk::RenderPass,
    ) -> Renderer {
        let fonts_binding = descriptor_set_layout_binding(
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            0,
            1,
            vk::ShaderStageFlags::FRAGMENT,
        );
        let bindings = [fonts_binding];
        let imgui_descriptor_set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        let imgui_descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&imgui_descriptor_set_layout_info, None)
                .unwrap()
        };
        let layouts = [imgui_descriptor_set_layout];
        let imgui_descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        let imgui_descriptor_set = unsafe {
            device
                .allocate_descriptor_sets(&imgui_descriptor_set_allocate_info)
                .unwrap()[0]
        };

        let mut fonts = imgui_ctx.fonts();
        let fonts_texture = fonts.build_rgba32_texture();
        let staging_buffer = create_buffer(
            &instance,
            physical_device,
            &device,
            fonts_texture.data.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data_ptr = device
                .map_memory(
                    staging_buffer.buffer_memory,
                    0,
                    fonts_texture.data.len() as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap() as *mut u8;
            data_ptr
                .copy_from_nonoverlapping(fonts_texture.data.as_ptr(), fonts_texture.data.len());
            device.unmap_memory(staging_buffer.buffer_memory);
        }

        let fonts_info = add_texture(
            instance,
            physical_device,
            device,
            command_pool,
            graphics_queue,
            vk::Extent3D {
                width: fonts_texture.width,
                height: fonts_texture.height,
                depth: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
            Some(staging_buffer),
            "fonts_texture",
            textures,
        );
        fonts.tex_id =
            imgui::TextureId::from(textures["fonts_texture"].image.image.as_raw() as usize);

        let infos = [fonts_info];
        let fonts_set_write = vk::WriteDescriptorSet::builder()
            .dst_set(imgui_descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&infos)
            .build();
        unsafe { device.update_descriptor_sets(&[fonts_set_write], &[]) };

        let imgui_pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder()
                        .set_layouts(&[imgui_descriptor_set_layout])
                        .push_constant_ranges(&[vk::PushConstantRange {
                            stage_flags: vk::ShaderStageFlags::VERTEX,
                            size: size_of::<[f32; 4]>() as u32,
                            ..Default::default()
                        }]),
                    None,
                )
                .unwrap()
        };

        let imgui_vert_shader = create_shader_module(
            &device,
            IMGUI_SHADER_SRC,
            "imgui",
            "vs_main",
            "vs_6_5",
            &shader_args,
            &vec![],
        );
        let imgui_frag_shader = create_shader_module(
            &device,
            IMGUI_SHADER_SRC,
            "imgui",
            "fs_main",
            "ps_6_5",
            &shader_args,
            &vec![],
        );

        pipeline_builder.pipeline_layout = imgui_pipeline_layout;

        pipeline_builder.shader_stages.clear();
        pipeline_builder
            .shader_stages
            .push(pipeline_shader_stage_create_info(
                vk::ShaderStageFlags::VERTEX,
                imgui_vert_shader,
                "vs_main\0",
            ));
        pipeline_builder
            .shader_stages
            .push(pipeline_shader_stage_create_info(
                vk::ShaderStageFlags::FRAGMENT,
                imgui_frag_shader,
                "fs_main\0",
            ));

        pipeline_builder.vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&[get_binding_description()])
            .vertex_attribute_descriptions(&get_attribute_description())
            .build();

        pipeline_builder.color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .build();

        pipeline_builder.depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default();

        pipeline_builder.dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR])
            .build();

        let imgui_pipeline = pipeline_builder.build_pipline(&render_pass, &device);

        unsafe {
            device.destroy_shader_module(imgui_vert_shader, None);
            device.destroy_shader_module(imgui_frag_shader, None);
        }

        let vert_buf = create_buffer(
            instance,
            physical_device,
            device,
            100 * 1024 * 1024,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        let idx_buf = create_buffer(
            instance,
            physical_device,
            device,
            100 * 1024 * 1024,
            vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        return Renderer {
            pipeline_layout: imgui_pipeline_layout,
            pipeline: imgui_pipeline,
            descriptor_set_layout: imgui_descriptor_set_layout,
            descriptor_set: imgui_descriptor_set,
            vert_buf,
            idx_buf,
        };
    }

    pub fn deinit_renderer(&mut self, device: &Device) {
        unsafe {
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            free_buffer_and_memory(device, &self.idx_buf);
            free_buffer_and_memory(device, &self.vert_buf);
            device.destroy_pipeline(self.pipeline, None);
        }
    }

    pub fn render(
        &self,
        draw_data: &imgui::DrawData,
        device: &Device,
        command_buffer: vk::CommandBuffer,
    ) {
        unsafe {
            let mut dst_idx_ptr = device
                .map_memory(
                    self.idx_buf.buffer_memory,
                    0,
                    (draw_data.total_idx_count as usize * size_of::<imgui::DrawIdx>()) as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap() as *mut imgui::DrawIdx;
            let mut dst_vert_ptr = device
                .map_memory(
                    self.vert_buf.buffer_memory,
                    0,
                    (draw_data.total_vtx_count as usize * size_of::<imgui::DrawVert>()) as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap() as *mut imgui::DrawVert;

            for draw_list in draw_data.draw_lists() {
                let idx_buf = draw_list.idx_buffer();
                dst_idx_ptr.copy_from_nonoverlapping(idx_buf.as_ptr(), idx_buf.len());
                dst_idx_ptr = dst_idx_ptr.offset(idx_buf.len() as isize);

                let vert_buf = draw_list.vtx_buffer();
                dst_vert_ptr.copy_from_nonoverlapping(vert_buf.as_ptr(), vert_buf.len());
                dst_vert_ptr = dst_vert_ptr.offset(vert_buf.len() as isize);
            }

            device.unmap_memory(self.idx_buf.buffer_memory);
            device.unmap_memory(self.vert_buf.buffer_memory);
        }

        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            let scale = [
                2.0 / draw_data.display_size[0],
                2.0 / draw_data.display_size[1],
            ];
            let translate = [
                -1.0 - draw_data.display_pos[0] * scale[0],
                -1.0 - draw_data.display_pos[1] * scale[1],
            ];
            device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                &[scale, translate].align_to::<u8>().1,
            );
            device.cmd_bind_index_buffer(
                command_buffer,
                self.idx_buf.buffer,
                0,
                vk::IndexType::UINT16,
            );
            device.cmd_bind_vertex_buffers(command_buffer, 0, &[self.vert_buf.buffer], &[0]);

            let fb_size = [
                draw_data.display_size[0] * draw_data.framebuffer_scale[0],
                draw_data.display_size[1] * draw_data.framebuffer_scale[1],
            ];

            device.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: draw_data.display_size[0],
                    height: draw_data.display_size[1],
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );

            let clip_off = draw_data.display_pos;
            let clip_scale = draw_data.framebuffer_scale;

            let (mut global_idx_offset, mut global_vert_offset) = (0, 0);
            for draw_list in draw_data.draw_lists() {
                for draw_cmd in draw_list.commands() {
                    match draw_cmd {
                        imgui::DrawCmd::Elements {
                            cmd_params: cmd,
                            count: idx_count,
                            ..
                        } => {
                            let clip_min = [
                                ((cmd.clip_rect[0] - clip_off[0]) * clip_scale[0]).max(0.0),
                                ((cmd.clip_rect[1] - clip_off[1]) * clip_scale[1]).max(0.0),
                            ];
                            let clip_max = [
                                ((cmd.clip_rect[2] - clip_off[0]) * clip_scale[0]).min(fb_size[0]),
                                ((cmd.clip_rect[3] - clip_off[1]) * clip_scale[1]).min(fb_size[1]),
                            ];
                            if clip_max[0] <= clip_min[0] || clip_max[1] <= clip_min[1] {
                                continue;
                            }

                            device.cmd_set_scissor(
                                command_buffer,
                                0,
                                &[vk::Rect2D {
                                    offset: vk::Offset2D {
                                        x: clip_min[0] as i32,
                                        y: clip_min[1] as i32,
                                    },
                                    extent: vk::Extent2D {
                                        width: (clip_max[0] - clip_min[0]) as u32,
                                        height: (clip_max[1] - clip_min[1]) as u32,
                                    },
                                }],
                            );

                            device.cmd_draw_indexed(
                                command_buffer,
                                idx_count as u32,
                                1,
                                (cmd.idx_offset + global_idx_offset) as u32,
                                (cmd.vtx_offset + global_vert_offset) as i32,
                                0,
                            );
                        }
                        _ => panic!("Yea, nah, mate, screw this"),
                    }
                }
                global_idx_offset += draw_list.idx_buffer().len();
                global_vert_offset += draw_list.vtx_buffer().len();
            }
        }
    }
}
