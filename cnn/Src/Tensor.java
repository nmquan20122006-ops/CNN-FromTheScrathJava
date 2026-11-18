package Src;
import java.io.Serializable;
/*
* Một ví dụ cho bài toán thể hiện cách mảng 1 chiều phối hợp với các component còn lại
*  cho 1 Src.Tensor: n=1 c=1 h=2 w=3
*  --> data tạo mảng chứa 6 phần tử "this.data = new double[n*c*h*w];"
*  --> _strides[0](của n) = 6 | _strides[1](của c) = 6 | _strides[2](của h) = 3 | _strides[3](của w) = 1
*  --> index = _strides[0]*6 + _strides[1]*6 + _strides[2]*3 + _strides[3]*1
*  Ví dụ mảng data chứa: ô0: 10 | ô1: 20 | ô2: 30 | ô3:40 | ô4:50 | ô5:60
*  Lấy pixel ở góc trên trái (0,0,0,0)
*  -> index 0*6 + 0*6 + 0*3 + 0*1 = 0 -> data[0] = 10
*  Dịch sang phải một cột cùng dòng (0,0,0,1)
*  -> index = 1
*  Dịch xuống dưới một dòng (0,0,1,0)
*  -> index = 3
* */
public class Tensor implements Serializable {
    private static final long serialVersionUID = 1L;

    public final double[] data; // đại diện cho các dữ liệu đầu vào (hình ảnh)
    public final int[] shape; // định nghĩa cấu trúc đa chiều của khối dữ liệu
    private final int[] _strides; // định nghĩa phải nhảy cóc bao nhiêu bước trên mảng 1 chiều
                                        //để tới được vị trí mong muốn
    /*
    * n: là batch size, khối dữ liệu này chứa bao nhiêu mẫu
    * c: là channels, số lượng kênh màu
    * h: là height, chiều cao ảnh
    * w: là width, chiều rộng ảnh
    *  */
    public Tensor(int n, int c,int h, int w){
        this.shape = new int[]{n,c,h,w};
        this.data = new double[n*c*h*w]; // số lượng phần tử cần thiết để chứa dữ liệu của mẫu
        this._strides = new int[]{c*h*w,h*w,w,1};
    }
    // tìm index phù hợp trong "data" với dữ liệu đầu vào
    private int index(int n, int c, int h, int w){
        return n*_strides[0]+c*_strides[1]+h*_strides[2]+w*_strides[3];
    }
    // lấy dữ liệu tại index
    public double get(int n, int c, int h, int w){
        return data[index(n,c,h,w)];
    }
    // gán giá trị tại index
    public void set(int n,int c,int h,int w, double v){
        data[index(n,c,h,w)]=v;
    }
    // các phương thức helper trả về shape của mẫu đang tính
    public int n(){return shape[0];}
    public int c(){return shape[1];}
    public int h(){return shape[2];}
    public int w(){return shape[3];}
    // trả về độ lớn của mẫu
    public int size(){return data.length;}
}
