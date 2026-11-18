package Layer;

import Src.Tensor;

public class Flatten implements Layer {
    private static final long serialVersionUID = 1L;

    private int[] _inputShape; // Lưu shape của lần forward gần nhất
    public Flatten(){}
    /*
    * Duỗi thẳng tất cả giá trị pixel / đặc trưng thành 1 vector 1D duy nhất cho mỗi batch
    * */
    @Override
    public Tensor forward(Tensor input) {
        int n = input.n(); // số lượng mẫu
        int c = input.c(); // số kênh
        int h = input.h();
        int w = input.w();

        int features = c*h*w; // tổng đặc trưng sau khi dàn phẳng
        _inputShape = input.shape.clone(); // lưu shape gốc
        Tensor out = new Tensor(n,features,1,1); // tạo out với số lượng kênh
        System.arraycopy(input.data,0,out.data,0,input.data.length);
        return out;
    }

    @Override
    public Tensor backward(Tensor dOut) {
        if(_inputShape==null) throw new IllegalArgumentException("Forward phai chay truoc");
        int n = _inputShape[0];
        int c = _inputShape[1];
        int h = _inputShape[2];
        int w = _inputShape[3];

        if(dOut.n()!=n||dOut.c()!=c*h*w||dOut.h()!=1||dOut.w()!=1){ // check shape đầu vào backward
            throw new IllegalArgumentException("Sai shape");
        }

        Tensor dInput = new Tensor(n,c,h,w);
        System.arraycopy(dOut.data,0, dInput.data,0,dOut.data.length);

        return dInput;
    }

    @Override
    public void update(double learningRate) {}

    @Override
    public void clearCache() {
        _inputShape = null;
    }
}
