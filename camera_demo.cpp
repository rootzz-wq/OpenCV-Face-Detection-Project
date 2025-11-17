#include <iostream>
#include <map>
#include <string>
#include <opencv2/opencv.hpp>


// 基于给定的模型文件构建深度神经网络，用于后面推理运算

// 加载人脸检测模型，该模型用于输入一帧图像，推理输出所有人脸区域位置坐标
cv::Ptr<cv::FaceDetectorYN> fd = cv::FaceDetectorYN::create("../../models/yunet.onnx", "", cv::Size(640, 480));

// 加载人脸识别模型，该模型用于输入一帧人脸图像，推理输出人脸特征向量(128 个点)
cv::Ptr<cv::FaceRecognizerSF> fr = cv::FaceRecognizerSF::create("../../models/face_recognizer_fast.onnx", "");

// 人脸识别的核心原理：比较两个人脸的特征向量的相似度（高于某个我们设置的阈值就认为是同一个人），基于余弦距离或欧式距离计算

// 存放人脸信息，相当于人脸数据库
std::map<std::string, cv::Mat> face_data;


void face_register();
std::string face_auth(cv::Mat img, cv::Mat face_box);


int main()
{
	cv::VideoCapture cap(0);

	if(!cap.isOpened())
	{
		std::cerr << "打开摄像头失败" << std::endl;
		return -1;
	}

	// 获取摄像头的属性，比如分辨率、帧率等
	std::cout << "分辨率：" << cap.get(cv::CAP_PROP_FRAME_WIDTH) << " * " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
	std::cout << "帧率：" << cap.get(cv::CAP_PROP_FPS) << std::endl;

	face_register();

	// 从摄像头读取一帧图像，使用 Mat 对象可以存放一帧图像的数据
	// Mat：Matrix，矩阵类
	cv::Mat img, faces;
	int keyValue, r;
	int x, y, w, h;
	std::string user_name;

	while(1)
	{
		// 从摄像头读取一帧图像，并将其存放在 img 对象中
		if(!cap.read(img))
		{
			std::cerr << "从摄像头读取图像失败" << std::endl;
			return -1;
		}	

		fd->setInputSize(img.size());
		// 输入一帧图像(img)给上面构建好的深度神经网络，它会进行推理运算得到输出结果（所有人脸区域的坐标位置数据）存放在 faces 对象中
		fd->detect(img, faces);

		// 每一行为一张人脸的坐标数据
		for(r = 0; r < faces.rows; r++)
		{
			// 从第 0 列到最后一列分别为：人脸区域的 x 坐标、y 坐标、宽度、高度

			// 置信度
			if(faces.at<float>(r, 14) < 0.9) continue;

			user_name = face_auth(img, faces.row(r));

			x = faces.at<float>(r, 0);
			y = faces.at<float>(r, 1);
			w = faces.at<float>(r, 2);
			h = faces.at<float>(r, 3);

			cv::rectangle(img, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(0, 255, 0), 2);	
			cv::putText(img, user_name, cv::Point(x, y - 10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);	
			

			// int right_eye_x, right_eye_y, left_eye_x, left_eye_y, nose_x, nose_y;

			// right_eye_x = faces.at<float>(r, 4);
			// right_eye_y = faces.at<float>(r, 5);
			// left_eye_x = faces.at<float>(r, 6);
			// left_eye_y = faces.at<float>(r, 7);
			// nose_x = faces.at<float>(r, 8);
			// nose_y = faces.at<float>(r, 9);

			// cv::circle(img, cv::Point(right_eye_x, right_eye_y), 30, cv::Scalar(0, 0, 0), 3);
			// cv::circle(img, cv::Point(left_eye_x, left_eye_y), 30, cv::Scalar(0, 0, 0), 3);

			// cv::circle(img, cv::Point(nose_x, nose_y), 30, cv::Scalar(0, 0, 255), -1);
		}

		// 转换颜色模式
		//cv::cvtColor(img, img,  cv::COLOR_BGR2RGB);

		// 遍历每一行
		// for (int row = 0; row < img.rows; ++row) {
		// 	// 获取当前行的指针
		// 	cv::Vec3b* ptr = img.ptr<cv::Vec3b>(row);
		// 	for (int col = 0; col < img.cols; ++col) {
		// 		// 操作像素，例如将每个像素的蓝色通道设为0
		// 		//ptr[col][2] = 0; // 蓝色通道设为0
		// 		// 同理，绿色通道为[1]，红色通道为[2]
		// 		//if(col < 320) continue;

		// 		// ptr[col][0] = 0;
		// 		// ptr[col][1] = 0;
		// 		// ptr[col][2] = 255;

		// 		// ptr[col][0] = 255 - ptr[col][0] ;
		// 		// ptr[col][1] = 255 - ptr[col][1] ;
		// 		// ptr[col][2] = 255 - ptr[col][2] ;
		// 	}
		// }		

		// 绘制线条
		// cv::line(img, cv::Point(0, 0), cv::Point(640, 480), cv::Scalar(255, 0, 0), 2);
		// cv::line(img, cv::Point(640, 0), cv::Point(0, 480), cv::Scalar(255, 255, 0), 2);

		// 绘制圆形
		// cv::circle(img, cv::Point(320, 240), 100, cv::Scalar(0, 0, 255), 5);

		// 绘制矩形
		// cv::rectangle(img, cv::Point(100, 100), cv::Point(300, 200), cv::Scalar(0, 255, 0), 3);

		// 绘制文本
		// cv::putText(img, "Jun ge yyds!", cv::Point(100, 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(100, 0, 100), 3);

		cv::imshow("junge yyds", img);  //  在名字为 junge yyds 的窗口上显示 img 图像

		keyValue = cv::waitKey(1);  // 等待用户按键，超时为 1ms，返回值为按键的编码值
		
		if(keyValue == 27) break;

		if(keyValue == ' ')
		{
			// 将当前帧图像保存为本地图片文件
			cv::imwrite("./zms.jpg", img);
		}
	}

	cap.release();  // 释放摄像头

	return 0;
}


// 人脸注册
void face_register()
{
    cv::Mat img, faces, aligned_face, face_feature;

    std::string face_imgs[] = {"zm", "mayun","junge"};

    for(const std::string& name : face_imgs)
    {
        img = cv::imread(name + ".jpg");
        
        // 检查图像是否成功加载
        if(img.empty())
        {
            std::cerr << "无法读取图片: " << name << ".jpg" << std::endl;
            continue;
        }

        fd->setInputSize(img.size());
        fd->detect(img, faces);

        if(faces.rows == 0)
        {
            std::cerr << "在图片 " << name << ".jpg 中没有检测到任何人脸！" << std::endl;
            continue;
        }

        if(faces.rows > 1)
        {
            std::cerr << "在图片 " << name << ".jpg 中检测到多张人脸！" << std::endl;
            continue;
        }

        // 将检测出人脸区域进行裁减和对齐处理
        fr->alignCrop(img, faces.row(0), aligned_face);

        // 将上面处理后的人脸图像输入到人脸识别模型，由它推理运算得到对应的人脸特征向量
        fr->feature(aligned_face, face_feature);

        face_data[name] = face_feature.clone();
    }
}


// 人脸验证
std::string face_auth(cv::Mat img, cv::Mat face_box)
{
	cv::Mat aligned_face, face_feature;
	double score;

	// 将检测出人脸区域进行裁减和对齐处理
	fr->alignCrop(img, face_box, aligned_face);

	// 将上面处理后的人脸图像输入到人脸识别模型，由它推理运算得到对应的人脸特征向量
	fr->feature(aligned_face, face_feature);
	
	for(auto it = face_data.begin(); it != face_data.end(); ++it)
	{
		score = fr->match(face_feature, it->second);  // 使用余弦距离比较相似度

		if(score > 0.363) return it->first;
	}

	return "unknow";
}
