//#include "stdafx.h"  
#include "cv.h"  
#include "highgui.h"  
#include "iostream"
#include "imgproc.hpp"
#define  u_char unsigned char  
#define  DIST 0.5  
#define  NUM 5                       //���������� ԭֵΪ20 �ٶ�̫�� ��С�����Ч����û�в��̫���ԣ����ٶ������ܶ�  

//ȫ�ֱ���  
bool pause = false;
bool is_tracking = false;
CvRect drawing_box;
IplImage *current;
double *hist1, *hist2;
double *m_wei;                                                                  //Ȩֵ����  
double C = 0.0;                                                                //��һ��ϵ��
int center_count = 0;														//���Ĺ켣�����
int old_t_x[10000], old_t_y[10000];                                 //���ĵ�

//�������ڱ�����Ƶ��ָ��
CvVideoWriter *writer = cvCreateVideoWriter("track.avi", CV_FOURCC('M', 'J', 'P', 'G'),29.9482, cvSize(640,360));

void init_target(double *hist1, double *m_wei, IplImage *current)
{
	IplImage *pic_hist = 0;
	int t_h, t_w, t_x, t_y;
	double h, dist;
	int i, j;
	int q_r, q_g, q_b, q_temp;

	t_h = drawing_box.height;
	t_w = drawing_box.width;
	t_x = drawing_box.x;
	t_y = drawing_box.y;

	h = pow(((double)t_w) / 2, 2) + pow(((double)t_h) / 2, 2);            //����  
	pic_hist = cvCreateImage(cvSize(300, 200), IPL_DEPTH_8U, 3);     //����ֱ��ͼͼ��  

																	 //��ʼ��Ȩֵ�����Ŀ��ֱ��ͼ  
	for (i = 0; i < t_w*t_h; i++)
	{
		m_wei[i] = 0.0;
	}

	for (i = 0; i<4096; i++)
	{
		hist1[i] = 0.0;
	}

	for (i = 0; i < t_h; i++)
	{
		for (j = 0; j < t_w; j++)
		{
			dist = pow(i - (double)t_h / 2, 2) + pow(j - (double)t_w / 2, 2);
			m_wei[i * t_w + j] = 1 - dist / h;                     //Ȩֵ�����ʼ����Ȩֵ������Ϊ�˺�����
			//printf("%f\n",m_wei[i * t_w + j]);  
			C += m_wei[i * t_w + j];
		}
	}

	//����Ŀ��Ȩֱֵ��  
	for (i = t_y; i < t_y + t_h; i++)
	{
		for (j = t_x; j < t_x + t_w; j++)
		{
			//rgb��ɫ�ռ�����Ϊ16*16*16 bins  
			q_r = ((u_char)current->imageData[i * current->widthStep + j * 3 + 2]) / 16;
			q_g = ((u_char)current->imageData[i * current->widthStep + j * 3 + 1]) / 16;
			q_b = ((u_char)current->imageData[i * current->widthStep + j * 3 + 0]) / 16;
			q_temp = q_r * 256 + q_g * 16 + q_b;
			hist1[q_temp] = hist1[q_temp] + m_wei[(i - t_y) * t_w + (j - t_x)];     //���ϸ����ش���Ȩֵ
		}
	}

	//��һ��ֱ��ͼ  
	for (i = 0; i<4096; i++)
	{
		hist1[i] = hist1[i] / C;
		//printf("%f\n",hist1[i]);  
	}

	//����Ŀ��ֱ��ͼ  
	double temp_max = 0.0;

	for (i = 0; i < 4096; i++)         //��ֱ��ͼ���ֵ��Ϊ�˹�һ��  
	{
		//printf("%f\n",val_hist[i]);  
		if (temp_max < hist1[i])
		{
			temp_max = hist1[i];
		}
	}
	//��ֱ��ͼ  
	CvPoint p1, p2;
	double bin_width = (double)pic_hist->width / 4096;
	double bin_unith = (double)pic_hist->height / temp_max;

	for (i = 0; i < 4096; i++)
	{
		p1.x = i * bin_width;
		p1.y = pic_hist->height;
		p2.x = (i + 1)*bin_width;
		p2.y = pic_hist->height - hist1[i] * bin_unith;
		//printf("%d,%d,%d,%d\n",p1.x,p1.y,p2.x,p2.y);  
		cvRectangle(pic_hist, p1, p2, cvScalar(0, 255, 0), -1, 8, 0);
	}
	cvSaveImage("hist1.jpg", pic_hist);
	cvReleaseImage(&pic_hist);
}

void MeanShift_Tracking(IplImage *current)
{
	int num = 0, i = 0, j = 0;
	int t_w = 0, t_h = 0, t_x = 0, t_y = 0;
	double *w = 0, *hist2 = 0;										//wΪ�����Ծ������飬w[i]Ϊÿ�����ص�ľ��� hist2Ϊ��ѡģ�͵�ֱ��ͼ����
	double sum_w = 0, x1 = 0, x2 = 0, y1 = 2.0, y2 = 2.0;			//sum_wΪ�����Ծ���֮�� y1Ϊ�����꣨��ֱ�����ϣ���ƫ�ƾ��� y2Ϊ�����꣨ˮƽ�����ϣ���ƫ�ƾ���
	int q_r, q_g, q_b;	
	int *q_temp;													//q_tempΪ����Ȥ�����������飬�洢�����ص�rgb��ɫ���ԣ�q_r * 256 + q_g * 16 + q_b��
	IplImage *pic_hist = 0;											//���ڻ�ֱ��ͼ

	t_w = drawing_box.width;
	t_h = drawing_box.height;

	pic_hist = cvCreateImage(cvSize(300, 200), IPL_DEPTH_8U, 3);     //����ֱ��ͼͼ��  
	hist2 = (double *)malloc(sizeof(double) * 4096);			     //��hist2ֱ��ͼ��������ڴ�
	w = (double *)malloc(sizeof(double) * 4096);				     //�������Ծ�����������ڴ�
	q_temp = (int *)malloc(sizeof(int)*t_w*t_h);					 //������Ȥ��������������������ڴ�

	//��¼��ÿ�����ĵ��λ���Ա���滭��
	old_t_x[center_count] = drawing_box.x+ t_w/2;
	old_t_y[center_count] = drawing_box.y+ t_h/2;
	//for (int ii = 0; ii < center_count; ii++)
	//{
	//	std::cout << old_t_x[ii] << std::endl;
	//}
	center_count += 1;

	while ((pow(y2, 2) + pow(y1, 2) > 0.5) && (num < NUM))           //������С��һ����ֵ
	{
		num++;
		t_x = drawing_box.x;
		t_y = drawing_box.y;
		memset(q_temp, 0, sizeof(int)*t_w*t_h);                     //��q_temp�����ʼ������
		for (i = 0; i<4096; i++)
		{
			w[i] = 0.0;
			hist2[i] = 0.0;
		}

		for (i = t_y; i < t_h + t_y; i++)
		{
			for (j = t_x; j < t_w + t_x; j++)
			{
				//rgb��ɫ�ռ�����Ϊ16*16*16 bins  
				q_r = ((u_char)current->imageData[i * current->widthStep + j * 3 + 2]) / 16;
				q_g = ((u_char)current->imageData[i * current->widthStep + j * 3 + 1]) / 16;
				q_b = ((u_char)current->imageData[i * current->widthStep + j * 3 + 0]) / 16;
				q_temp[(i - t_y) *t_w + j - t_x] = q_r * 256 + q_g * 16 + q_b;
				hist2[q_temp[(i - t_y) *t_w + j - t_x]] = hist2[q_temp[(i - t_y) *t_w + j - t_x]] + m_wei[(i - t_y) * t_w + j - t_x];
			}
		}

		//��һ��ֱ��ͼ  
		for (i = 0; i<4096; i++)
		{
			hist2[i] = hist2[i] / C;
			//printf("%f\n",hist2[i]);  
		}
		//����Ŀ��ֱ��ͼ  
		double temp_max = 0.0;

		for (i = 0; i<4096; i++)         //��ֱ��ͼ���ֵ��Ϊ�˹�һ��  
		{
			if (temp_max < hist2[i])
			{
				temp_max = hist2[i];
			}
		}
		//��ֱ��ͼ  
		CvPoint p1, p2;
		double bin_width = (double)pic_hist->width / (4368);
		double bin_unith = (double)pic_hist->height / temp_max;

		for (i = 0; i < 4096; i++)
		{
			p1.x = i * bin_width;
			p1.y = pic_hist->height;
			p2.x = (i + 1)*bin_width;
			p2.y = pic_hist->height - hist2[i] * bin_unith;
			cvRectangle(pic_hist, p1, p2, cvScalar(0, 255, 0), -1, 8, 0);
		}
		cvSaveImage("hist2.jpg", pic_hist);

		for (i = 0; i < 4096; i++)						//�����Ծ������w[i]Ϊÿ�����ص�ľ���
		{
			if (hist2[i] != 0)
			{
				w[i] = sqrt(hist1[i] / hist2[i]);           //�ݶ�
			}
			else
			{
				w[i] = 0;
			}
		}

		sum_w = 0.0;
		x1 = 0.0;
		x2 = 0.0;

		for (i = 0; i < t_h; i++)
		{
			for (j = 0; j < t_w; j++)
			{
				//printf("%d\n",q_temp[i * t_w + j]);  
				sum_w = sum_w + w[q_temp[i * t_w + j]];
				x1 = x1 + w[q_temp[i * t_w + j]] * (i - t_h / 2);
				x2 = x2 + w[q_temp[i * t_w + j]] * (j - t_w / 2);
			}
		}
		y1 = x1 / sum_w;					
		y2 = x2 / sum_w;

		//���ĵ�λ�ø���  
		drawing_box.x += y2;
		drawing_box.y += y1;

		//printf("%d,%d\n",drawing_box.x,drawing_box.y);  
	}
	free(hist2);
	free(w);
	free(q_temp);
	//��ʾ���ٽ��  
	cvRectangle(current, cvPoint(drawing_box.x, drawing_box.y), cvPoint(drawing_box.x + drawing_box.width, drawing_box.y + drawing_box.height), CV_RGB(255, 0, 0), 2);

//  �������ĵ�켣��ÿ�ζ������ĵ�켣���飬��ͷ��ʼ����ÿһ֡��ͷ��
	for (int o = 0; o < center_count -1; o++)
	{
		cvLine(current, cvPoint(old_t_x[o], old_t_y[o]), cvPoint(old_t_x[o+1], old_t_y[o+1]), CV_RGB(249, 205, 173), 3);  //����
	}

	//������ϸ��ٿ��Լ���¼�켣�ߵ�ͼ��д������Ƶtrack.avi
	cvWriteFrame(writer, current);

	cvShowImage("Meanshift", current);
//	cvWaitKey(0);
	//cvSaveImage("result.jpg",current);  
	cvReleaseImage(&pic_hist);
}

void onMouse(int event, int x, int y, int flags, void *param)
{
	if (pause)
	{
		switch (event)
		{
		case CV_EVENT_LBUTTONDOWN:
			//the left up point of the rect  
			drawing_box.x = x;
			drawing_box.y = y;
			break;
		case CV_EVENT_LBUTTONUP:
			//finish drawing the rect (use color green for finish)  
			drawing_box.width = x - drawing_box.x;
			drawing_box.height = y - drawing_box.y;
			cvRectangle(current, cvPoint(drawing_box.x, drawing_box.y), cvPoint(drawing_box.x + drawing_box.width, drawing_box.y + drawing_box.height), CV_RGB(255, 0, 0), 2);
			cvShowImage("Meanshift", current);

			//Ŀ���ʼ��  
			hist1 = (double *)malloc(sizeof(double) * 16 * 16 * 16);
			m_wei = (double *)malloc(sizeof(double)*drawing_box.height*drawing_box.width);
			init_target(hist1, m_wei, current);
			is_tracking = true;

			//����ѡȡĿ�꣬���֮ǰ�ĺۼ��������¼������̧���пɱ��������뻭�浼����������켣
			for (int i = 0; i < center_count + 10; i++)  //���֮ǰ���ĵ�켣�����еĴ洢��ֵ
			{
				old_t_x[i] = NULL;
				old_t_y[i] = NULL;
			}
			center_count = 0;                 //���¿�ʼ����

			break;
		}
		return;
	}
}



int main()
{
	CvCapture *capture = cvCreateFileCapture("5.mp4");
	current = cvQueryFrame(capture);
	char res[20];
	int nframe = 0;
	int count = 0;   //֡����


	//���ڻ�ȡ������Ƶ����Ϣ��֡��fps,��a,��b��,ÿ�θ�����Ƶ����Ҫ���¼��㣬C170������fps=29.9482,a=640,b=360(cvSize(640,360))
	//double fps = cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);
	////��cvSize��������һ��CvSize���͵ı���size�����Ⱥ͸߶���������Ƶ�ļ���ͬ  
	//int a = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
	//int b =	(int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);

	//std::cout << "fps="<< fps << std::endl;
	//std::cout << "a=" << a << std::endl;
	//std::cout << "b=" << b << std::endl;

	while (1)
	{
		/*  sprintf(res,"result%d.jpg",nframe);
		cvSaveImage(res,current);
		nframe++;*/
		if (is_tracking)
		{
			MeanShift_Tracking(current);
		}

		int c = cvWaitKey(1);
		//��ͣ  
		if (c == 'p')
		{
			pause = true;
			cvSetMouseCallback("Meanshift", onMouse, 0);
		}
		while (pause) {
			if (cvWaitKey(0) == 'p')
				pause = false;
		}
		cvShowImage("Meanshift", current);
		current = cvQueryFrame(capture); //ץȡһ֡  

		count += 1;
		std::cout << count << std::endl;
	}

	cvNamedWindow("Meanshift", 1);
	cvReleaseCapture(&capture);
	cvDestroyWindow("Meanshift");
	cvReleaseVideoWriter(&writer);
	cvReleaseImage(&current);
	return 0;
}