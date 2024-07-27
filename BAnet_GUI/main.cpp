#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Native_File_Chooser.H>
#include <FL/Fl_Progress.H>
#include <FL/fl_ask.H>
#include <FL/Fl_Check_Button.H>

#include <filesystem>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

const size_t img_size = 640;
const size_t channels = 3;

void click_textbox(Fl_Widget*, void*);
void classify(Fl_Widget*, void*);
void loadimage(std::string, cv::Mat&);
void readlabels(std::string, std::vector<std::string>&);

std::string input_path;
std::string output_path;

struct Paths {
	std::wstring model_path;
	std::string label_path;
	Fl_Input* target;
	Fl_Input* save_target;
	Fl_Check_Button* usingGPU;
};

int main(int argc, char** argv) {


	Fl_Window* window = new Fl_Window(600, 400);

	// 각각의 텍스트들
	const char* input_label = u8"분류할 경로 입력";
	const char* run_label = u8"실행";
	const char* output_label = u8"저장할 경로 입력";
	const char* btn_label = u8"폴더 선택";
	const char* pregress_label = u8"진행도";

	// 경로들
	std::wstring model_path = L"./model.onnx";
	std::string label_path = "./labels.txt";

	// 분류할 경로를 입력받는 것을 표시하는 라벨
	Fl_Box* input_box = new Fl_Box(20, 40, 200, 100);
	input_box->box(FL_FLAT_BOX);
	input_box->label(input_label);
	input_box->labelfont(FL_FREE_FONT);
	input_box->labelsize(15);
	input_box->labeltype(FL_NORMAL_LABEL);

	// 분류된 경로를 입력받는 것을 표시하는 라벨
	Fl_Box* output_box = new Fl_Box(20, 160, 200, 100);
	output_box->box(FL_FLAT_BOX);
	output_box->label(output_label);
	output_box->labelfont(FL_FREE_FONT);
	output_box->labelsize(15);
	output_box->labeltype(FL_NORMAL_LABEL);

	// 분류할 경로를 입력받음
	Fl_Input* input_text = new Fl_Input(250, 80, 200, 30);
	
	// 경로 입력 버튼
	Fl_Button* input_btn = new Fl_Button(500, 80, 60, 30, btn_label);
	input_btn->callback(click_textbox, input_text);

	// 저장할 경로를 입력받음
	Fl_Input* output_text = new Fl_Input(250, 200, 200, 30);


	// 경로 입력 버튼
	Fl_Button* output_btn = new Fl_Button(500, 200, 60, 30, btn_label);
	output_btn->callback(click_textbox, output_text);

	// 프로그레스 바
	Fl_Progress* progress = new Fl_Progress(100, 310, 150, 30, pregress_label);

	// GPU 사용 여부 라디오 버튼
	Fl_Check_Button* usingGPUbtn = new Fl_Check_Button(450, 310, 100, 50, u8"CUDA사용");

	// 실행 버튼
	Fl_Button* run = new Fl_Button(300, 300, 100, 50, run_label);
	run->labelsize(15);


	// 이벤트 등록
	Paths paths = { model_path, label_path,input_text, output_text, usingGPUbtn };
	run->callback(classify, &paths);

	window->end();
	window->show();
	return Fl::run();
}

/* 파일 선택 */
void click_textbox(Fl_Widget* w, void* param) {
	Fl_Input* i = reinterpret_cast<Fl_Input*>(param);
	const char* title = u8"폴더 선택";
	Fl_Native_File_Chooser* chooser = new Fl_Native_File_Chooser(Fl_Native_File_Chooser::BROWSE_DIRECTORY);
	chooser->title(title);

	switch (chooser->show()) {
		case -1: i->value("error"); break;
		case 1: break;
		default: i->value(chooser->filename()); input_path = chooser->filename();
	}
}

/* 분류 */
void classify(Fl_Widget* w, void* params) {
	Paths* contents = reinterpret_cast<Paths*>(params);

	// 타겟 경로 가져오기
	Fl_Input* target_input = reinterpret_cast<Fl_Input*>(contents->target);
	const char* target_path = target_input->value();

	// 저장할 경로 가져오기
	Fl_Input* save_input = reinterpret_cast<Fl_Input*>(contents->save_target);
	std::string save_path = save_input->value();

	Ort::Env env;
	Ort::SessionOptions options;

	if (contents->usingGPU->value() == '1') {
		OrtCUDAProviderOptions cudaOption;
		cudaOption.device_id = 0;
		options.AppendExecutionProvider_CUDA(cudaOption); // CUDA 사용 설정
	}

	Ort::Session* session;

	try {
		session = new Ort::Session(env, contents->model_path.c_str(), options);; // 모델 불러오기
	}
	catch (const Ort::Exception& e) {
		fl_alert(u8"모델을 찾을 수 없습니다.");
		std::cout << e.what();
		return;
	}
	/*
	std::vector<std::string> labels;
	readlabels(contents->label_path, labels);
	*/


	Ort::AllocatorWithDefaultOptions allocator; // allocator

	auto metaDatas = session->GetModelMetadata();
	auto customMetaDataKeys = metaDatas.GetCustomMetadataMapKeysAllocated(allocator);
	std::string names(metaDatas.LookupCustomMetadataMapAllocated("names", allocator).get());
	std::cout << names << "\n";

	clock_t total_start_time = clock();

	for (const auto& file : std::filesystem::recursive_directory_iterator(input_path, std::filesystem::directory_options::skip_permission_denied)) {

		auto start_time = clock();

		if (file.is_directory() || !file.exists()) continue;

		std::string path = file.path().string();

		std::cout << path << "\n";


		cv::Mat input;
		loadimage(path.c_str(), input);

		// Input 개수
		size_t num_input_nodes = session->GetInputCount();
		std::vector<const char*> input_node_names(num_input_nodes);
		std::vector<int64_t> input_node_dims;
		std::vector<Ort::AllocatedStringPtr> input_node_name_allocated_strings;

		// Input node 차원 정보 넣기
		for (size_t i = 0; i < num_input_nodes; ++i) {
			auto input_name = session->GetInputNameAllocated(i, allocator);
			input_node_name_allocated_strings.push_back(std::move(input_name));
			input_node_names.push_back(input_node_name_allocated_strings.back().get());
		}

		size_t input_tensor_size = input.total();

		std::cout << input_tensor_size << "\n";


		// 데이터를 실수 벡터로 복사
		std::vector<float> input_tensor_values(input_tensor_size);
		input_tensor_values.assign(input.begin<float>(), input.end<float>());

		// 입력텐서에 데이터 넣기
		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), input_node_dims.size());

		assert(input_tensor.IsTensor());

		size_t num_output_nodes = session->GetOutputCount();
		std::vector<const char*> output_names;
		std::vector<Ort::AllocatedStringPtr> output_node_name_allocated_strings;
		for (size_t i = 0; i < num_output_nodes; ++i) {
			auto name = session->GetOutputNameAllocated(i, allocator);
			output_node_name_allocated_strings.push_back(std::move(name));
			output_names.push_back(output_node_name_allocated_strings.back().get());
		}

		std::vector<const char*> output_node_names = { output_names.front() };

		// 세션 실행, 텐서 벡터 형태로 리턴
		auto output_tensors = session->Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), output_node_names.size());
		assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

		std::cout << "Output:" << output_tensors.data() << "\n";
		

		std::cout << "Time:" << (clock() - start_time) / 1000.0 << "\n";
	}

	delete session;
	std::cout << (clock() - total_start_time) / 1000.0<< "\n";
}

void loadimage(std::string path, cv::Mat& img) {
	auto inputImage = cv::imread(path); // 이미지 읽기

	cv::Size dnnInputSize = cv::Size(640, 640);;
	cv::Scalar mean = cv::Scalar(0.485, 0.456, 0.406);
	bool swapRB = true;

	cv::Mat blob;
	// ONNX: (N x 3 x H x W)
	cv::dnn::blobFromImage(inputImage, img, 1.0 / 255.0, dnnInputSize, mean, swapRB, false);
}

void readlabels(std::string path, std::vector<std::string>& container) {
	std::ifstream file(path);

	std::string line;

	while (std::getline(file, line)) {
		container.push_back(line);
	}
}

