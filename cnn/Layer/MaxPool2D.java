package Layer;
import Src.Tensor;

public class MaxPool2D implements Layer {
    private static final long serialVersionUID = 1L;

    private final int _size; // kích thước của cửa sổ pooling
    private final int _stride; // bước nhảy
    private int[] _argmax; // với mỗi ô output, lưu flat index của phần tử lớn nhất trong input
    private int[] _inputShape; // lưu kích thước gốc của tensor đầu vào
    private int _outH, _outW; // lưu kích thước output để dùng lại lúc backward

    public MaxPool2D(int size){
        this(size,size);
    }

    public MaxPool2D(int size, int stride) {
        if(size<=0) throw new IllegalArgumentException("Pool phai lon hon 0");
        if(stride<=0) throw new IllegalArgumentException("Stride phai lon hon 0");
        this._size = size;
        this._stride = stride;
    }

    @Override
    public Tensor forward(Tensor input) {
        int n = input.n();
        int c = input.c();
        int h = input.h();
        int w = input.w();

        int outH = (h-_size)/_stride+1;
        int outW = (w-_size)/_stride+1;
        if(outH<=0||outW<=0) throw new IllegalArgumentException("dau vao qua nho cho pool size nay");

        Tensor out = new Tensor(n,c,outH,outW);
        // mỗi ô output cần nhớ 1 vị trí argmax trong input
        _argmax = new int[n*c*outH*outW];

        _inputShape = input.shape.clone();
        _outH = outH;
        _outW = outW;
        for(int ni=0;ni<n;ni++){ // chọn bức ảnh thứ mấy trong batch
            for(int ci=0;ci<c;ci++){ // chọn kênh màu/bản đồ đặc trưng thứ mấy
                for(int oh=0;oh<outH;oh++){ // duyệt chiều cao ảnh mới
                    for(int ow=0;ow<outW;ow++){ // duyệt chiều rộng ảnh mới

                        int startH = oh*_stride; // bắt đầu tại chiều cao ảnh mới
                        int startW = ow*_stride; // bắt đầu tại chiều rộng ảnh mới

                        double maxVal = Double.NEGATIVE_INFINITY;
                        int maxFlatIndex = -1;
                        //dùng filter (ih,iw) quét qua ô nằm trong mảng pooling
                        for(int ih=startH;ih<startH+_size;ih++){
                            for(int iw=startW;iw<startW+_size;iw++){
                                double v = input.get(ni,ci,ih,iw);
                                if(v>maxVal){
                                    maxVal = v;
                                    maxFlatIndex = flatIndex(ni,ci,ih,iw,c,h,w);
                                }
                            }
                        }
                        out.set(ni,ci,oh,ow,maxVal); // set maxVal cho out
                        // lưu vị trí argmax vào đúng ô output tương ứng
                        _argmax[flatIndex(ni,ci,oh,ow,c,outH,outW)] = maxFlatIndex;
                    }
                }
            }
        }
        return out;
    }

    @Override
    public Tensor backward(Tensor dOut) {
        if(_argmax==null||_inputShape==null){
            throw new IllegalArgumentException("Forward phai chay truoc");
        }
        int n = _inputShape[0];
        int c = _inputShape[1];
        int h = _inputShape[2];
        int w = _inputShape[3];
        // gradient nhận vào phải khớp shape của output lúc forward
        if(dOut.n()!=n||dOut.c()!=c||dOut.h()!=_outH||dOut.w()!=_outW){
            throw new IllegalArgumentException("dOut khong khop shape cua output");
        }

        Tensor dInput = new Tensor(n,c,h,w); // data khởi tạo mặc định toàn 0
        for(int ni=0;ni<n;ni++){ // duyệt từng ảnh trong batch
            for(int ci=0;ci<c;ci++){ // duyệt từng kênh
                for(int oh=0;oh<_outH;oh++){ // duyệt từng ô output
                    for(int ow=0;ow<_outW;ow++){
                        int outFlat = flatIndex(ni,ci,oh,ow,c,_outH,_outW);
                        int inFlat = _argmax[outFlat]; // vị trí max đã lưu lúc forward
                        // gradient chỉ chảy về đúng phần tử thắng max,
                        // các phần tử khác trong cửa sổ nhận 0.
                        // dùng += vì các cửa sổ có thể chồng nhau khi stride < size
                        dInput.data[inFlat] += dOut.data[outFlat];
                    }
                }
            }
        }
        return dInput;
    }
    @Override
    public void update(double learningRate) {}

    @Override
    public void clearCache() {
        _argmax = null;
        _inputShape = null;
    }

    private int flatIndex(int ni,int ci, int ih, int iw, int C, int H, int W){
        return ni*C*H*W + ci*H*W + ih*W + iw;
    }
}
