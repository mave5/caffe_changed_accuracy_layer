#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe {
	template <typename Dtype>
	void AccuracyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*> & bottom, const vector<Blob<Dtype>*>& top) {
		//top_k_ = this->layer_param_.accuracy_param().top_k();
	  	has_ignore_label_ = this->layer_param_.accuracy_param().has_ignore_label();
		if (has_ignore_label_) {
		    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
		}
}


template <typename Dtype>
void AccuracyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	//CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count()) << "top_k must be less than or equal to the number of classes.";


	label_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
	outer_num_ = bottom[0]->count(0, label_axis_); // number of images
	inner_num_ = bottom[0]->count(label_axis_ + 1); // number of channels, here equal 1

	/*CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      	<< "Number of labels must match number of predictions; "
      	<< "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      	<< "label count (number of labels) must be N*H*W, "
      	<< "with integer values in {0, 1, ..., C-1}.";*/
  	
	vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  	top[0]->Reshape(top_shape);
  	if (top.size() > 1) {
    	// Per-class accuracy is a vector; 1 axes.
    	vector<int> top_shape_per_class(1);
    	top_shape_per_class[0] = bottom[0]->shape(label_axis_);
    	top[1]->Reshape(top_shape_per_class);
    	nums_buffer_.Reshape(top_shape_per_class);
  }
}


template <typename Dtype>
void AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) {
	Dtype accuracy = 0;
	
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bottom_label = bottom[1]->cpu_data();
  	const int dim = bottom[0]->count() / outer_num_; // network output size
	const int num_labels = bottom[0]-> shape(label_axis_); // label size, should be equal to output size

	//vector<Dtype> maxval(top_k_+1);
  	//vector<int> max_id(top_k_+1);
  	
	
	int count = 0; 
	int A_and_B, A_one,B_one; // these are for calculating Dice metric= 2(A and B)/(A or B) 
	float dice[outer_num_];	
	int bin_out[dim], bin_label[dim]; // output and ground truth values 
  	
	for (int i = 0; i < outer_num_; ++i){ // loop over number of images
		A_and_B=0;A_one=0;B_one=0; // set these to zero for each image
    		for (int j = 0; j < inner_num_; ++j) { // loop over channels, here equal 1
			for (int k = 0; k < num_labels; ++k) {// loop over output/label size
		       		bin_out[k]=bottom_data[i * dim + k * inner_num_ + j]>=.5; // convert network output to binary value 0/1  
				bin_label[k]=bottom_label[i * dim + k * inner_num_ + j]; // ground truth values

				if (bin_out[k]+bin_label[k]==2) A_and_B ++; //cacular intersection of two sets
				if (bin_out[k]==1) A_one++; // count number of 1s in A
				if (bin_label[k]==1) B_one++;// count number of 1s in B
		       }
      		}
		//std::cout<< "A and B="<< A_and_B<<std::endl;
		//std::cout<< "A or B ="<< A_one+B_one<<std::endl;
		dice[i]=(2*A_and_B)/float(A_one+B_one);
		//std::cout<< "Dice="<< dice[i]<<std::endl;
                accuracy+=dice[i];
      		++count;
    	}
  	
    
  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]-> mutable_cpu_data()[0] = accuracy / (count);

}

INSTANTIATE_CLASS(AccuracyLayer);
REGISTER_LAYER_CLASS(Accuracy);

}  // namespace caffe

