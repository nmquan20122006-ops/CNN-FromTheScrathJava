package Layer;
import Src.Tensor;
/*
* ReLU là một cổng cho phép tín hiệu dương đi qua còn tín hiệu âm thì chặn lại
* --> tạo ra tính phi tuyến cho mạng và giúp mạng học nhanh hơn
* */
public class ReLU implements Layer {
    private static final long serialVersionUID = 1L;

    private double[] _mask; // lưu những ô lúc forward là dương, âm
    @Override
    public Tensor forward(Tensor input) { // -> tính ReLU và ghi nhớ mask
        // tạo một tensor đầu ra cùng shape với input
        Tensor out = new Tensor(input.n(), input.c(), input.h(), input.w());
        _mask = new double[input.data.length]; // cấp phát mảng mask với kích thước bằng tensor
        for(int i=0;i<input.data.length;i++){ // duyệt toàn bộ mảng input
            if(input.data[i]>0){ // nếu giá trị > 0
                out.data[i] = input.data[i]; // gán giá trị input bằng với out
                _mask[i] = 1.0; // đánh dấu mask tại i = 1
            }else{ // bé hơn 0
                out.data[i] = 0.0; // gán giá trị mask = 0
                _mask[i] = 0.0; // đánh dấu mask = 1
            }
        }
        return out; // trả out cho layer khác
    }
    @Override
    public Tensor backward(Tensor dOut) {
        Tensor dInput = new Tensor(dOut.n(),dOut.c(),dOut.h(),dOut.w());
        for(int i=0;i<dOut.data.length;i++){
            dInput.data[i] = dOut.data[i]*_mask[i];
        }
        return dInput;
    }
    @Override
    public void update(double learningRate) {

    }

    @Override
    public void clearCache() {
        _mask = null;
    }
}
