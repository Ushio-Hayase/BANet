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
#include <locale>
#include <string>
#include <codecvt>
#include <iostream>
#include <fstream>

const size_t img_size = 640;
const size_t channels = 3;

void click_textbox(Fl_Widget*, void*);
void classify(Fl_Widget*, void*);
int loadimage(std::filesystem::path, cv::Mat&);
void readlabels(std::string, std::vector<std::wstring>&);
std::size_t number_of_files_in_directory(std::filesystem::path);
void infer(void*, Fl_Button*);


struct Paths {
	std::string model_path;
	std::string label_path;
	Fl_Input* target;
	Fl_Input* save_target;
	Fl_Check_Button* usingGPU;
	Fl_Progress* progress;
};

int APIENTRY WinMain(HINSTANCE hInstance,
	HINSTANCE hPrevInstance,
	LPSTR     lpCmdLine,
	int       nCmdShow)
{
	Fl_Window* window = new Fl_Window(600, 400);

	// ������ �ؽ�Ʈ��
	const char* input_label = u8"�з��� ��� �Է�";
	const char* run_label = u8"����";
	const char* output_label = u8"������ ��� �Է�";
	const char* btn_label = u8"���� ����";
	const char* pregress_label = u8"���൵";

	// ��ε�
	std::string model_path = "./model.onnx";
	std::string label_path = "./labels.txt";

	// �з��� ��θ� �Է¹޴� ���� ǥ���ϴ� ��
	Fl_Box* input_box = new Fl_Box(20, 40, 200, 100);
	input_box->box(FL_FLAT_BOX);
	input_box->label(input_label);
	input_box->labelfont(FL_HELVETICA);
	input_box->labelsize(15);
	input_box->labeltype(FL_NORMAL_LABEL);

	// �з��� ��θ� �Է¹޴� ���� ǥ���ϴ� ��
	Fl_Box* output_box = new Fl_Box(20, 160, 200, 100);
	output_box->box(FL_FLAT_BOX);
	output_box->label(output_label);
	output_box->labelfont(FL_HELVETICA);
	output_box->labelsize(15);
	output_box->labeltype(FL_NORMAL_LABEL);

	// �з��� ��θ� �Է¹���
	Fl_Input* input_text = new Fl_Input(250, 80, 200, 30);
	input_text->value("");

	// ��� �Է� ��ư
	Fl_Button* input_btn = new Fl_Button(500, 80, 60, 30, btn_label);
	input_btn->callback(click_textbox, input_text);

	// ������ ��θ� �Է¹���
	Fl_Input* output_text = new Fl_Input(250, 200, 200, 30);
	output_text->value("");

	// ��� �Է� ��ư
	Fl_Button* output_btn = new Fl_Button(500, 200, 60, 30, btn_label);
	output_btn->callback(click_textbox, output_text);

	// ���α׷��� ��
	Fl_Progress* progress = new Fl_Progress(100, 310, 150, 30, pregress_label);

	// GPU ��� ���� ���� ��ư
	Fl_Check_Button* usingGPUbtn = new Fl_Check_Button(450, 310, 100, 50, u8"CUDA���");

	// ���� ��ư
	Fl_Button* run = new Fl_Button(300, 300, 100, 50, run_label);
	run->labelsize(15);


	// �̺�Ʈ ���
	Paths paths = { model_path, label_path,input_text, output_text, usingGPUbtn, progress };
	run->callback(classify, &paths);

	window->end();
	window->show();
	return Fl::run();
}

/* ���� ���� */
void click_textbox(Fl_Widget* w, void* param) {
	Fl_Input* i = reinterpret_cast<Fl_Input*>(param);
	const char* title = u8"���� ����";
	Fl_Native_File_Chooser* chooser = new Fl_Native_File_Chooser(Fl_Native_File_Chooser::BROWSE_DIRECTORY);
	chooser->title(title);

	switch (chooser->show()) {
	case -1: i->value("error"); break;
	case 1: break;
	default: i->value(chooser->filename());
	}
}

/* �з� */
void classify(Fl_Widget* w, void* params) {
	Paths* contents = reinterpret_cast<Paths*>(params);

	Fl_Button* btn = reinterpret_cast<Fl_Button*>(w);
	btn->label(u8"������");
	btn->redraw_label();
	btn->value(1);
	btn->deactivate();


	std::thread p(infer, params, btn);
	p.detach();
}

// ������ 1 ��ȯ, ���н� 0 ��ȯ
int loadimage(std::filesystem::path path, cv::Mat& img) {
	std::ifstream input(path, std::ios::binary);

	std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});

	auto inputImage = cv::imdecode(buffer, cv::IMREAD_COLOR); // �̹��� �б�

	cv::Size dnnInputSize = cv::Size(640, 640);;
	cv::Scalar mean = cv::Scalar(0, 0, 0);

	cv::Mat blob;
	// ONNX: (N x 3 x H x W)
	try {
		cv::dnn::blobFromImage(inputImage, img, 1.0 / 255.0, dnnInputSize, mean, true, false);
		return 1;
	}
	catch (cv::Exception& e) {
		std::cerr << e.msg << "\n";
		return 0;
	}
}

void readlabels(std::string path, std::vector<std::wstring>& container) {
	std::locale::global(std::locale(".UTF-8"));

	std::wifstream file(path);

	std::wstring line;

	while (std::getline(file, line)) {
		container.push_back(line);
	}
}

std::size_t number_of_files_in_directory(std::filesystem::path path) {
	using std::filesystem::directory_iterator;
	return std::distance(directory_iterator(path), directory_iterator{});
}

void infer(void* params, Fl_Button* btn) {

	Paths* contents = reinterpret_cast<Paths*>(params);

	// Ÿ�� ��� ��������
	Fl_Input* target_input = reinterpret_cast<Fl_Input*>(contents->target);
	std::string input_path = target_input->value();

	// ������ ��� ��������
	Fl_Input* save_input = reinterpret_cast<Fl_Input*>(contents->save_target);
	std::string save_path_a = save_input->value();
	std::wstring save_path;
	save_path.assign(save_path_a.begin(), save_path_a.end());

	if (save_path == L"" || input_path == "") {
		Fl::lock();
		btn->label(u8"����");
		btn->value(0);
		btn->redraw_label();
		btn->activate();
		Fl::unlock();
		return;
	}

	size_t totalFileCount = number_of_files_in_directory(input_path); // ���� ����

	// ����� ��
	Fl_Progress* progressbar = reinterpret_cast<Fl_Progress*>(contents->progress);
	progressbar->maximum(totalFileCount);
	progressbar->value(0.0);

	cv::dnn::Net model;

	try {
		model = cv::dnn::readNetFromONNX(contents->model_path); // �� �ҷ�����
	}
	catch (const cv::Exception& e) {
		fl_alert(u8"���� ã�� �� �����ϴ�.");
		std::cout << e.what();
		return;
	}

	if (contents->usingGPU->value() == 1) {
		model.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		model.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}

	std::vector<std::wstring> labels;
	readlabels(contents->label_path, labels); // �� �ҷ�����

	size_t processing = 0; // �����

	clock_t total_start_time = clock();

	if (!std::filesystem::exists(save_path + L"\\��Ÿ")) std::filesystem::create_directory(save_path + L"\\��Ÿ");

	for (const auto& file : std::filesystem::recursive_directory_iterator(input_path, std::filesystem::directory_options::skip_permission_denied)) {

		auto start_time = clock();

		if (file.is_directory() || !file.exists()) {
			Fl::lock();
			progressbar->value(++processing);
			Fl::unlock();
			continue;
		}

		auto path = file.path();
		auto fileName = file.path().filename().wstring();

		cv::Mat input;
		if (!loadimage(path, input)) continue; // �̹��� �ε� �� blob���� ����

		model.setInput(input); // �� �Է� ����

		cv::Mat output;
		model.forward(output); // �� ����

		cv::Point claasIdPoint;
		double max_prob;
		cv::minMaxLoc(output, nullptr, &max_prob, nullptr, &claasIdPoint);

		int classId = claasIdPoint.x; // �з��� �ε���

		auto target_path = save_path + L"\\" + labels[classId];




		if (max_prob > 0.3 && !std::filesystem::exists(std::filesystem::path(target_path + L"\\" + fileName))) {
			if (!std::filesystem::exists(target_path)) std::filesystem::create_directory(target_path); // ���� ���� �� ����
			std::filesystem::copy_file(path, target_path + L"\\" + fileName);
		}
		else if (!std::filesystem::exists(save_path + L"\\��Ÿ\\" + fileName)) std::filesystem::copy_file(path, save_path + L"\\��Ÿ\\" + fileName);

		Fl::lock();
		progressbar->value(++processing);
		Fl::unlock();
	}

	Fl::lock();
	btn->label(u8"����");
	btn->value(0);
	btn->redraw_label();
	btn->activate();
	Fl::unlock();
}