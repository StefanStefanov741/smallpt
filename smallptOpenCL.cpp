//g++ -std=c++11 -O3 -fopenmp smallptOpenCL.cpp -o smallptOpenCL -lOpenCL

#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <CL/cl2.hpp>

using namespace std;
using namespace cl;

//Global OpenCL variables
cl_float3* canvas;
CommandQueue queue;
Device device;
Kernel kernel;
Context context;
Program program;
Buffer cl_output;
Buffer cl_spheres;
Buffer cl_samples;

//Material reflection types (diffuse,specular,refractive)
enum Refl_t { 
  DIFF,
  SPEC,
  REFR
};  

struct Sphere
{
	cl_float radius;
	cl_float3 position;
	cl_float3 color;
	cl_float3 emission;
	Refl_t refl;

	Sphere(float rad, cl_float3 p, cl_float3 e, cl_float3 c, Refl_t refl) :
    	radius(rad), position(p), emission(e), color(c), refl(refl) {}
};

void initOpenCL()
{
	//Get the first available OpenCL platform
	std::vector<Platform> platforms;
	Platform::get(&platforms);
	//Pick one platform
	Platform platform = platforms[0];

	//Get the first available OpenCL device (My Nvidia GPU)
	std::vector<Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	device = devices[0];
	
	//Create an OpenCL context and command queue for the gpu
	context = Context(device);
	queue = CommandQueue(context, device);

	//Read the kernel file code
	string source;
	ifstream file("kernel.cl");
	while (!file.eof()){
		char line[256];
		file.getline(line, 255);
		source += line;
	}

	const char* kernel_source = source.c_str();

	//Create OpenCL program from the kernel code
	program = Program(context, kernel_source);
	program.build({ device });

	//Create a kernel object that executes the render function from the new program
	kernel = Kernel(program, "render");
}

inline double clamp(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
inline int toInt(double x) { return int(pow(clamp(x), 1/2.2) * 255 + .5); }
#define float3(x, y, z) {{x, y, z}};

//Scene with spheres
Sphere spheres[] = {
  //radius, position, emission, color, material
  Sphere(200, {-200.52f, 0.0f, 0.0f}, {0,0,0}, {.75,.25,.25}, DIFF), //Left
  Sphere(200, {200.52f, 0.0f, 0.0f}, {0,0,0}, {.25,.25,.75}, DIFF), //Right
  Sphere(200, {0.0f, 0.0f, -200.94f}, {0,0,0}, {.75,.75,.75}, DIFF), //Back
  Sphere(200, {0.0f, 0.0f, 202.0f}, {0,0,0}, {0,0,0}, DIFF), //Front
  Sphere(200, {0.0f, 200.4f, 0.0f}, {0,0,0}, {.75,.75,.75}, DIFF), //Top
  Sphere(200, {0.0f, -200.47f, 0.0f}, {0,0,0}, {.75,.75,.75}, DIFF), //Bottom
  Sphere(0.17, {-0.24f, -0.29f, -0.5f}, {0,0,0}, {1,1,1}, SPEC), //Mirror
  Sphere(0.17, {0.24f, -0.29f, -0.28f}, {0,0,0}, {1,1,1}, REFR), //Glass
  Sphere(1, {0.0f, 1.382f, -0.25f}, {12,12,12}, {0,0,0}, DIFF) //Light
};

int main(int argc, char *argv[]){
	int width = 1024;
	int height = 768;
	//Get samples input
	int samps = argc == 2 ? atoi(argv[1]) / 4 : 1;
	const int spheres_count = 9;

	//Initialize OpenCL
	initOpenCL();

	//Create final canvas
	canvas = new cl_float3[width * height];

	//Create buffers on the OpenCL device for the image and the scene
	cl_output = Buffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(cl_float3));
	cl_spheres = Buffer(context, CL_MEM_READ_ONLY, spheres_count * sizeof(Sphere));
	cl_samples = Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
	queue.enqueueWriteBuffer(cl_spheres, CL_TRUE, 0, spheres_count * sizeof(Sphere), spheres);
	queue.enqueueWriteBuffer(cl_samples, CL_TRUE, 0, sizeof(int), &samps);

	//Specify OpenCL kernel arguments
	kernel.setArg(0, cl_spheres);
	kernel.setArg(1, width);
	kernel.setArg(2, height);
	kernel.setArg(3, spheres_count);
	kernel.setArg(4, cl_output);
	kernel.setArg(5, samps);

	//Every pixel has its own work item
	std::size_t global_size = width * height;
	//Set the local size to the maximum number of work items for one group for my gpu
	std::size_t local_size = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

	//Launch the kernel
	queue.enqueueNDRangeKernel(kernel, NULL, global_size, local_size);
	queue.finish();

	//Read and copy OpenCL output to CPU
	queue.enqueueReadBuffer(cl_output, CL_TRUE, 0, width * height * sizeof(cl_float3), canvas);

	//Save image
	FILE *f = fopen("image.ppm", "w");
	fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
	for (int i = 0; i < width * height; i++){
		fprintf(f, "%d %d %d ",
		toInt(canvas[i].s[0]),
		toInt(canvas[i].s[1]),
		toInt(canvas[i].s[2]));
	}

	//free memory
	delete canvas;

	return 0;
}
