package Layer;

import Src.Adam;
import Src.Tensor;

import java.util.Arrays;
import java.util.Random;

public class Conv2D implements Layer{
    private static final long serialVersionUID = 1L;

    //                  Cấu hình layer
    private final int _inChannels; // số kênh của ảnh/bản đồ đặc trưng vào
    private final int _outChannels; // số lượng filter / feature map
    private final int _kernelSize; // kích thuớc kernel quét
    private final int _stride; // bảo kernel phải dịch bao nhiêu ô
    private final int _padding; // số vòng 0 thêm quanh ảnh trước khi quét nhằm giữ kích thước

    //                  Tham số mạng học được
    private final double[] _weights; // toàn bộ trọng số của các filter
    private final double[] _bias; // mỗi filter có 1 bias

    //                  Gradient
    private final double[] _dWeights; // lưu đạo hàm loss/weight lúc backward
    private final double[] _dBias; // lưu đạo hàm loss/bias lúc backward

    private Tensor _input; // tensor đầu vào
    private int _outH,_outW; // heigh, weight đã tính lúc lan truyền tiến

    // Adam lưu trạng thái riêng cho weights và bias; null nghĩa là dùng SGD
    private Adam _adamWeights;
    private Adam _adamBias;

    public Conv2D(int inChannels, int outChannels, int kernelSize){
        this(inChannels,outChannels,kernelSize,1,0);
    }
    public Conv2D(int inChannels, int outChannels, int kernelSize, int stride, int padding){
        if(inChannels<=0) throw new IllegalArgumentException("inChannels phai lon hon 0");
        if(outChannels<=0) throw new IllegalArgumentException("outChannels phai lon hon 0");
        if(kernelSize<=0) throw new IllegalArgumentException("kernelSize phai lon hon 0");
        if(stride<=0) throw new IllegalArgumentException("stride phai lon hon 0");
        if(padding<0) throw new IllegalArgumentException("padding khong duoc am");

        this._inChannels = inChannels;
        this._outChannels = outChannels;
        this._kernelSize = kernelSize;
        this._stride = stride;
        this._padding = padding;

        int perFilter = inChannels*kernelSize*kernelSize; // trọng số của một filter
        this._weights = new double[outChannels*perFilter]; // tổng trọng số của tất cả filter
        this._bias = new double[outChannels]; // tổng bias cho từng filter
        this._dWeights = new double[outChannels*perFilter];
        this._dBias = new double[outChannels];

        initWeights();
    }

    private void initWeights(){
        Random rnd = new Random(42);
        double fanIn = _inChannels * _kernelSize * _kernelSize; // tổng số kết nối đầu vào
        double std = Math.sqrt(2.0/fanIn); // độ lệch chuẩn
        for(int i=0;i<_weights.length;i++){
            _weights[i] = rnd.nextGaussian()*std; // gán trọng số ban đầu là
        }
    }
    @Override
    public Tensor forward(Tensor input) {
        int n = input.n();
        int c = input.c();
        int h = input.h();
        int w = input.w();
        if(c!=_inChannels){throw new IllegalArgumentException("So kenh input khong khop!");}
        //Công thức tính đầu ra của feature map
        int outH = (h+2*_padding-_kernelSize)/_stride+1;
        int outW = (w+2*_padding-_kernelSize)/_stride+1;
        if(outH<=0||outW<=0) throw new IllegalArgumentException("dau vao qua nho cho kernel size nay");
        Tensor out = new Tensor(n,_outChannels,outH,outW); // Lưu ý tensor mới này chỉ mới có kích thước chưa có data
        _input = input;
        _outH = outH; // chiều cao đầu ra (feature map)
        _outW = outW; // chiều rộng đầu ra (feature map)

        for(int ni=0;ni<n;ni++){ // Duyệt ảnh vào lần lượt
            for(int oc=0;oc<_outChannels;oc++){ // Lần lượt dùng bộ lọc để quét ảnh

                // mục tiêu của vòng lặp này để đổ dữ liệu vào bên trong feature map
                for(int oh=0;oh<_outH;oh++){ // duyệt tọa độ hàng trên feature map
                    for(int ow=0;ow<_outW;ow++){ // duyệt tọa độ cột trên feature map

                        double sum = _bias[oc];
                        for(int ic=0;ic<_inChannels;ic++){ // Duyệt từng kênh đầu vào
                            for(int kh=0;kh<_kernelSize;kh++){
                                for(int kw=0;kw<_kernelSize;kw++){

                                    int ih = oh*_stride+kh-_padding; // tọa đồ hàng ảnh đầu vào
                                    int iw = ow*_stride+kw-_padding; // tọa độ cột ảnh đầu vào

                                    if(ih<0||ih>=h||iw<0||iw>=w) continue;
                                    // sum = lấy giá trị pixel ở ảnh gốc * trọng số filter
                                    sum+=input.get(ni,ic,ih,iw)*_weights[weightIndex(oc,ic,kh,kw)];
                                }
                            }
                        }
                        out.set(ni,oc,oh,ow,sum);
                    }
                }
            }
        }
        return out;
    }

    private int weightIndex(int oc, int ic, int kh, int kw)
    { return ((oc * _inChannels + ic) * _kernelSize + kh) * _kernelSize + kw;}

    @Override
    public Tensor backward(Tensor dOut) {
        if(_input==null){throw new IllegalArgumentException("Forward phai chay truoc");}
        int n = _input.n();
        int c = _input.c();
        int h = _input.h();
        int w = _input.w();
        if(dOut.n()!=n||dOut.c()!=_outChannels||dOut.h()!=_outH||dOut.w()!=_outW){
            throw new IllegalArgumentException("dOut khong khop shape cua output");
        }
        zeroGrad();
        Tensor dInput = new Tensor(n,c,h,w);

        for(int ni=0;ni<n;ni++){
            for(int oc=0;oc<_outChannels;oc++){
                for(int oh=0;oh<_outH;oh++){
                    for(int ow=0;ow<_outW;ow++){
                        double grad = dOut.get(ni,oc,oh,ow);
                        if(grad==0) continue;
                        _dBias[oc]+=grad;

                        for(int ic=0;ic<_inChannels;ic++){
                            for(int kh=0;kh<_kernelSize;kh++){
                                for(int kw=0;kw<_kernelSize;kw++){

                                    int ih = oh*_stride-_padding+kh;
                                    int iw = ow*_stride-_padding+kw;

                                    if(ih<0||ih>=h||iw<0||iw>=w)continue;

                                    int wi = weightIndex(oc,ic,kh,kw);
                                    double inVal = _input.get(ni,ic,ih,iw);

                                    _dWeights[wi] += grad * inVal; // dW += dOut * input

                                    int di = flatIndex(ni,ic,ih,iw,c,h,w);
                                    dInput.data[di] += grad * _weights[wi];
                                }
                            }
                        }
                    }
                }
            }
        }
        return dInput;
    }

    @Override
    public void setOptimizer(String name) {
        if ("adam".equals(name)) {
            _adamWeights = new Adam(_weights.length);
            _adamBias = new Adam(_bias.length);
        } else if ("sgd".equals(name)) {
            _adamWeights = null;
            _adamBias = null;
        } else {
            throw new IllegalArgumentException("Optimizer khong hop le: " + name);
        }
    }

    @Override
    public void update(double learningRate) {
        if (_adamWeights != null) {
            _adamWeights.update(_weights, _dWeights, learningRate);
            _adamBias.update(_bias, _dBias, learningRate);
            return;
        }
        for(int i=0;i<_weights.length;i++){
            _weights[i] -= learningRate*_dWeights[i];
        }
        for(int oc=0;oc<_bias.length;oc++){
            _bias[oc] -=learningRate*_dBias[oc];
        }
    }

    @Override
    public void clearCache() {
        _input = null;
        _adamWeights = null; // không lưu trạng thái optimizer vào file model
        _adamBias = null;
    }

    private void zeroGrad(){
        Arrays.fill(_dWeights,0.0);
        Arrays.fill(_dBias,0.0);
    }
    private int flatIndex(int ni, int ci, int ih, int iw, int C, int H, int W) {
        return ni * C * H * W + ci * H * W + ih * W + iw;
    }

    // ===== Dùng cho test/debug: trả về bản sao để không sửa được tham số thật =====
    public double[] weights() {
        return _weights.clone();
    }

    public double[] bias() {
        return _bias.clone();
    }

    public double[] dWeights() {
        return _dWeights.clone();
    }

    public double[] dBias() {
        return _dBias.clone();
    }
}
