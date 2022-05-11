#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

//CMP3752M Parallel Progarmming
//Jack Davis 19701851
// 
//this code was created using workshops 2 and 3 as a base with the my_kernels code from workshop 3 then exapnded on.
//This code was developed to digtally enchance an image, it first starts by making a histogram to find the values of the pixles of a given
//image, this was done with atomic numbers. next was a cumlative histogram which mesaures the value of the pixles against the pixles themselves
//next it was time to normalise the cumlative histogram, this was done by using a scan pattern, namely a look-up table(LUT) to help map the pixles
//on the output image. lastly was to put the new valuees from the loot onto the output image and them display it. in the host code i outputed
//the run time and addtional information for each step as well as the result for each step in the command line when the program is executed.
//the code in the kernal file is basic but runs effectively for the small files we are using, however the code is limited and does not
//allow the user to select thhings like the bin size and may in turn struggle with extermly large files.

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input, "input");

		//a 3x3 convolution mask implementing an averaging 
		std::vector<float> convolution_mask = { 1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9 };

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 4 - device operations

		typedef int mytype;
		std::vector<mytype> H(256);
		std::vector<mytype> CUMLATIVEH(256);
		std::vector<mytype> LUT(256);
		size_t histsize = H.size() * sizeof(mytype);

		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size());
		cl::Buffer dev_Hist(context, CL_MEM_READ_WRITE, histsize);
		cl::Buffer dev_Cumlative_Hist(context, CL_MEM_READ_WRITE, histsize);
		cl::Buffer dev_LUT(context, CL_MEM_READ_WRITE, histsize);
//		cl::Buffer dev_convolution_mask(context, CL_MEM_READ_ONLY, convolution_mask.size()*sizeof(float));

		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		queue.enqueueFillBuffer(dev_Hist, 0, 0, histsize);
		//		queue.enqueueWriteBuffer(dev_convolution_mask, CL_TRUE, 0, convolution_mask.size()*sizeof(float), &convolution_mask[0]);
		
		//4.2 Setup and execute the kernel (i.e. device code)
		//histagram code
		cl::Kernel kernel_hist = cl::Kernel(program, "hist_image");
		kernel_hist.setArg(0, dev_image_input);
		kernel_hist.setArg(1, dev_Hist);
		//kernel.setArg(2, dev_convolution_mask);

		cl::Event prof_event;

		queue.enqueueNDRangeKernel(kernel_hist, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event);
		queue.enqueueReadBuffer(dev_Hist, CL_TRUE, 0, histsize, &H[0]);

		queue.enqueueFillBuffer(dev_Cumlative_Hist, 0, 0, histsize);

		//cumlative histagram
		cl::Kernel Culmative_hist = cl::Kernel(program, "Cumlative_hs");
		Culmative_hist.setArg(0, dev_Hist);
		Culmative_hist.setArg(1, dev_Cumlative_Hist);

		cl::Event prof_event2;

		queue.enqueueNDRangeKernel(Culmative_hist, cl::NullRange, cl::NDRange(histsize), cl::NullRange, NULL, &prof_event2);
		queue.enqueueReadBuffer(dev_Cumlative_Hist, CL_TRUE, 0, histsize, &CUMLATIVEH[0]);

		queue.enqueueFillBuffer(dev_LUT, 0, 0, histsize);

		//LUT
		cl::Kernel KLUT = cl::Kernel(program, "LUT");
		KLUT.setArg(0, dev_Cumlative_Hist);
		KLUT.setArg(1, dev_LUT);

		cl::Event prof_event3;

		queue.enqueueNDRangeKernel(KLUT, cl::NullRange, cl::NDRange(histsize), cl::NullRange, NULL, &prof_event3);
		queue.enqueueReadBuffer(dev_LUT, CL_TRUE, 0, histsize, &LUT[0]);

		//image output
		cl::Kernel Back_Projecttion = cl::Kernel(program, "Back_Projecttion");
		Back_Projecttion.setArg(0, dev_image_input);
		Back_Projecttion.setArg(1, dev_LUT);
		Back_Projecttion.setArg(2, dev_image_output);

		cl::Event prof_event4;

		vector<unsigned char> output_buffer(image_input.size());
		queue.enqueueNDRangeKernel(Back_Projecttion, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event4);
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		//4.3 Copy the result from device to host

		cout << endl;
		std::cout << "Histogram = " << H << std::endl;
		std::cout << "Hitogram execution time[ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Full Info: " << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << std::endl;

		cout << endl;
		std::cout << "Cumlative Histogram = " << CUMLATIVEH << std::endl;
		std::cout << "Cumlative Hitogram execution time[ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event2.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Full Info: " << GetFullProfilingInfo(prof_event2, ProfilingResolution::PROF_US) << std::endl;

		cout << endl;
		std::cout << "LUT = " << LUT << std::endl;
		std::cout << "LUT execution time[ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event3.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Full Info: " << GetFullProfilingInfo(prof_event3, ProfilingResolution::PROF_US) << std::endl;

		cout << endl;
		std::cout << "Vector execution time[ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event4.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Full Info: " << GetFullProfilingInfo(prof_event4, ProfilingResolution::PROF_US) << std::endl;

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image, "output");

		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
