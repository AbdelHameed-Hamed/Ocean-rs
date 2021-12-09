# ToDo

- Move Camera into its own file.
- Get the skybox shader to work with arbitrary resolutions.
- Maybe switch out the cooley-tukey FFT with a Stockham implementation?
- Smooth the camera movement.
- Add better camera controls (e.g., only rotate when you're holding Ctrl + mouse movement).
- Figure out why the fragment shader outputs very pixelated(?) colors and fix that.
- Support window resizing + fullscreen.
- Support shader hot-reloading.
- Check Empirical directional wave spectra for computer graphics for better, more realistic waves.
- Add realistic shading of the ocean surface.
- Perhaps replace the simple skybox with something that simulates the sky based on Rayleigh scattering?

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