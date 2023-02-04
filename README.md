# Demo

https://user-images.githubusercontent.com/18620914/161441943-ed51c5f8-a553-4a30-844f-d97dc705c74f.mp4

# Building

First, you need to have a few things installed.
- [Rust](https://www.rust-lang.org/)
- [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/)
- SDL2 if you're on Linux.

To build just use `cargo build` in the root directory.

# ToDo

- [ ] Maybe switch out the cooley-tukey FFT with a Stockham implementation?
- [ ] Refactor SceneData cbuffer in shaders.
- [ ] Smooth the camera movement.
- [ ] Add better camera controls (e.g., only rotate when you're holding Ctrl + mouse movement).
- [ ] Figure out why the fragment shader outputs very pixelated(?) colors and fix that.
- [ ] Support window resizing + fullscreen.
- [ ] Support shader hot-reloading.
- [x] Check Empirical directional wave spectra for computer graphics for better, more realistic waves.
- [ ] Add realistic shading of the ocean surface.
- [ ] Perhaps replace the simple skybox with something that simulates the sky based on Rayleigh scattering?
- [ ] Perhaps add a way to specify what the pipeline will look like from the shaders?
- ~[ ] Maybe do physical camera as is done [here](https://bitsquid.blogspot.com/2017/09/physical-cameras-in-stingray.html)?~

# References

## Papers
- [Simulating Ocean Water - Jerry Tessendorf](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.161.9102&rep=rep1&type=pdf)
- [Ocean Surface Generation and Rendering - Thomas Gamper](https://www.cg.tuwien.ac.at/research/publications/2018/GAMPER-2018-OSG/GAMPER-2018-OSG-thesis.pdf)
- [Empirical Directional Wave Spectra for Computer Graphics - Christopher J. Horvath](https://dl.acm.org/doi/10.1145/2791261.2791267)
- [Realtime GPGPU FFT Ocean Water Simulation - Fynn-Jorin Flügge](https://tore.tuhh.de/handle/11420/1439?locale=en)

## Repositories
- [FFT-Ocean](https://github.com/gasgiant/FFT-Ocean)
- [Oreon Engine](https://github.com/fynnfluegge/oreon-engine)
- [Asylum Tutorials](https://github.com/asylum2010/Asylum_Tutorials)

## Videos
- [Ocean waves simulation with Fast Fourier transform](https://youtu.be/kGEqaX4Y4bQ)
- [OpenGL FFT Ocean Water Tutorial](https://youtu.be/B3YOLg0sA2g)
